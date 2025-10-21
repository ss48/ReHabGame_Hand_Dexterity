// Copyright (c) 2023 homuler
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

using System.Collections;
using System.Collections.Generic;
using Mediapipe.Tasks.Vision.HandLandmarker;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

namespace Mediapipe.Unity.Sample.HandLandmarkDetection
{
    public class HandLandmarkerRunner : VisionTaskApiRunner<HandLandmarker>
    {
        [SerializeField] private HandLandmarkerResultAnnotationController _handLandmarkerResultAnnotationController;

        [SerializeField] private GameObject _landmarkPrefab;
        [SerializeField] private float _spacing = 15;
        [SerializeField] private bool _flipY = true;

        [SerializeField] private Vector3 _leftHandRefPos, _rightHandRefPos;

        private Experimental.TextureFramePool _textureFramePool;
        private GameObject[] _landmarkObjects;
        bool _isStale = false;
        bool _objectsHidden = false;
        float _lastUpdateTime = -10;
        private const int _objPoolSize = 42;

        private List<List<Vector3>> _handLandmarks;
        private List<bool> _handIsRight;
        private object _landmarkLock = new object();

        public readonly HandLandmarkDetectionConfig config = new HandLandmarkDetectionConfig();

        private void Awake()
        {
            _landmarkObjects = new GameObject[_objPoolSize];
            for (int i = 0; i < _objPoolSize; i++)
            {
                _landmarkObjects[i] = Instantiate(_landmarkPrefab, transform);
                _landmarkObjects[i].name = $"Landmark_{i}";
                _landmarkObjects[i].SetActive(false);
            }
        }


        public override void Stop()
        {
            base.Stop();
            _textureFramePool?.Dispose();
            _textureFramePool = null;
        }


        protected override IEnumerator Run()
        {
            Debug.Log($"Delegate = {config.Delegate}");
            Debug.Log($"Image Read Mode = {config.ImageReadMode}");
            Debug.Log($"Running Mode = {config.RunningMode}");
            Debug.Log($"NumHands = {config.NumHands}");
            Debug.Log($"MinHandDetectionConfidence = {config.MinHandDetectionConfidence}");
            Debug.Log($"MinHandPresenceConfidence = {config.MinHandPresenceConfidence}");
            Debug.Log($"MinTrackingConfidence = {config.MinTrackingConfidence}");

            yield return AssetLoader.PrepareAssetAsync(config.ModelPath);

            var options = config.GetHandLandmarkerOptions(config.RunningMode == Tasks.Vision.Core.RunningMode.LIVE_STREAM ? OnHandLandmarkDetectionOutput : null);
            taskApi = HandLandmarker.CreateFromOptions(options, GpuManager.GpuResources);
            var imageSource = ImageSourceProvider.ImageSource;

            yield return imageSource.Play();

            if (!imageSource.isPrepared)
            {
                Debug.LogError("Failed to start ImageSource, exiting...");
                yield break;
            }

            // Use RGBA32 as the input format.
            // TODO: When using GpuBuffer, MediaPipe assumes that the input format is BGRA, so maybe the following code needs to be fixed.
            _textureFramePool = new Experimental.TextureFramePool(imageSource.textureWidth, imageSource.textureHeight, TextureFormat.RGBA32, 10);

            // NOTE: The screen will be resized later, keeping the aspect ratio.
            screen.Initialize(imageSource);

            SetupAnnotationController(_handLandmarkerResultAnnotationController, imageSource);

            var transformationOptions = imageSource.GetTransformationOptions();
            var flipHorizontally = transformationOptions.flipHorizontally;
            var flipVertically = transformationOptions.flipVertically;
            var imageProcessingOptions = new Tasks.Vision.Core.ImageProcessingOptions(rotationDegrees: (int)transformationOptions.rotationAngle);

            AsyncGPUReadbackRequest req = default;
            var waitUntilReqDone = new WaitUntil(() => req.done);
            var waitForEndOfFrame = new WaitForEndOfFrame();
            var result = HandLandmarkerResult.Alloc(options.numHands);

            // NOTE: we can share the GL context of the render thread with MediaPipe (for now, only on Android)
            var canUseGpuImage = SystemInfo.graphicsDeviceType == GraphicsDeviceType.OpenGLES3 && GpuManager.GpuResources != null;
            using var glContext = canUseGpuImage ? GpuManager.GetGlContext() : null;

            while (true)
            {
                if (isPaused)
                {
                    yield return new WaitWhile(() => isPaused);
                }

                if (!_textureFramePool.TryGetTextureFrame(out var textureFrame))
                {
                    yield return new WaitForEndOfFrame();
                    continue;
                }

                // Build the input Image
                Image image;
                switch (config.ImageReadMode)
                {

                    case ImageReadMode.GPU:
                        if (!canUseGpuImage)
                        {
                            throw new System.Exception("ImageReadMode.GPU is not supported");
                        }
                        textureFrame.ReadTextureOnGPU(imageSource.GetCurrentTexture(), flipHorizontally, flipVertically);
                        image = textureFrame.BuildGPUImage(glContext);
                        // TODO: Currently we wait here for one frame to make sure the texture is fully copied to the TextureFrame before sending it to MediaPipe.
                        // This usually works but is not guaranteed. Find a proper way to do this. See: https://github.com/homuler/MediaPipeUnityPlugin/pull/1311
                        yield return waitForEndOfFrame;
                        break;

                    case ImageReadMode.CPU:
                        yield return waitForEndOfFrame;
                        textureFrame.ReadTextureOnCPU(imageSource.GetCurrentTexture(), flipHorizontally, flipVertically);
                        image = textureFrame.BuildCPUImage();
                        textureFrame.Release();
                        break;

                    case ImageReadMode.CPUAsync:
                    default:
                        req = textureFrame.ReadTextureAsync(imageSource.GetCurrentTexture(), flipHorizontally, flipVertically);
                        yield return waitUntilReqDone;

                        if (req.hasError)
                        {
                            Debug.LogWarning($"Failed to read texture from the image source");
                            continue;
                        }
                        image = textureFrame.BuildCPUImage();
                        textureFrame.Release();
                        break;
                }


                switch (taskApi.runningMode)
                {

                    case Tasks.Vision.Core.RunningMode.IMAGE:
                        if (taskApi.TryDetect(image, imageProcessingOptions, ref result))
                        {
                            _handLandmarkerResultAnnotationController.DrawNow(result);
                        }
                        else
                        {
                            _handLandmarkerResultAnnotationController.DrawNow(default);
                        }
                        break;

                    case Tasks.Vision.Core.RunningMode.VIDEO:
                        if (taskApi.TryDetectForVideo(image, GetCurrentTimestampMillisec(), imageProcessingOptions, ref result))
                        {
                            _handLandmarkerResultAnnotationController.DrawNow(result);
                        }
                        else
                        {
                            _handLandmarkerResultAnnotationController.DrawNow(default);
                        }
                        break;

                    case Tasks.Vision.Core.RunningMode.LIVE_STREAM:
                        taskApi.DetectAsync(image, GetCurrentTimestampMillisec(), imageProcessingOptions);
                        break;
                }
            }
        }


        private void OnHandLandmarkDetectionOutput(HandLandmarkerResult result, Image image, long timestamp)
        {
            _handLandmarkerResultAnnotationController.DrawLater(result);

            if (result.handLandmarks == null) return;

            int handCount = result.handWorldLandmarks.Count;

            lock (_landmarkLock)
            {
                _handLandmarks = new List<List<Vector3>>();
                _handIsRight = new List<bool>();
                for (int i = 0; i < handCount; ++i)
                {
                    _handLandmarks.Add(new List<Vector3>());

                    string categoryName = result.handedness[i].categories[0].categoryName;
                    _handIsRight.Add(categoryName == "Right");

                    for (int j = 0; j < result.handWorldLandmarks[i].landmarks.Count; ++j)
                    {
                        var landmark = result.handWorldLandmarks[i].landmarks[j];
                        Vector3 pos = new Vector3(landmark.x, landmark.y, landmark.z);
                        _handLandmarks[i].Add(pos);

                    }
                }
                _isStale = true;
            }
        }


        private void LateUpdate()
        {
            if (_isStale)
            {
                RefreshLandmarkObjects();
                _lastUpdateTime = Time.time;
                _objectsHidden = false;
                _isStale = false;
            }
            else if (Time.time > _lastUpdateTime + 0.5f)
            {
                HideLandmarkObjects();
                _lastUpdateTime = Time.time + 1000f;
                _objectsHidden = true;
            }
        }


        private void RefreshLandmarkObjects()
        {
            lock (_landmarkLock)
            {
                int handCount = _handLandmarks.Count;
                int objCount = 0;
                for (int i = 0; i < handCount; ++i)
                {
                    for (int j = 0; j < _handLandmarks[i].Count; ++j)
                    {
                        var obj = _landmarkObjects[objCount];
                        obj.SetActive(true);
                        obj.transform.position = _handLandmarks[i][j] * _spacing;
                        if(_flipY)
                        {
                            obj.transform.position = new Vector3(obj.transform.position.x, -obj.transform.position.y, obj.transform.position.z);
                        }

                        if (_handIsRight[i])
                        {
                            obj.transform.position += _rightHandRefPos;
                        }
                        else
                        {
                            obj.transform.position += _leftHandRefPos;
                        }
                        ++objCount;
                    }
                }

                for (int i = objCount; i < _objPoolSize; ++i)
                {
                    _landmarkObjects[i].SetActive(false);
                }
            }
        }


        private void HideLandmarkObjects()
        {
            for (int i = 0; i < _objPoolSize; ++i)
            {
                _landmarkObjects[i].SetActive(false);
            }
        }


#if UNITY_EDITOR
        private void OnGUI()
        {
            if (_objectsHidden) return;

            for (int i = 0; i < _landmarkObjects.Length; ++i)
            {
                if (_landmarkObjects[i].activeInHierarchy)
                {
                    Handles.Label(_landmarkObjects[i].transform.position, i.ToString());
                }
            }
        }
#endif
    }
}
