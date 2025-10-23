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

        [SerializeField] private GameObject _palmPrefab, _fingerPrefab, _nodePrefab;
        [SerializeField] private float _spacing = 15;
        [SerializeField] private bool _flipY = true;

        [SerializeField] private bool _mirror = true;

        [Tooltip("Factor to multiply the 2D image offset into 3d movement. Effectively the play area.")]
        [SerializeField] private float _handOffsetMagnitude;

        private Experimental.TextureFramePool _textureFramePool;
        private List<List<HandObject>> _handObjects;
        bool _isStale = false;
        bool _objectsHidden = false;
        float _lastUpdateTime = -10;

        private List<List<Vector3>> _handLandmarks;
        private List<bool> _handIsRight;
        private object _landmarkLock = new object();

        public readonly HandLandmarkDetectionConfig config = new HandLandmarkDetectionConfig();

        private void Awake()
        {
            _handObjects = new();

            List<HandObject> leftHand = new();
            leftHand.Add(new PalmObject(_palmPrefab, 0, 5, 17));

            leftHand.Add(new FingerObject(_fingerPrefab, 1, 2));
            leftHand.Add(new FingerObject(_fingerPrefab, 3, 4));

            leftHand.Add(new FingerObject(_fingerPrefab, 5, 6));
            leftHand.Add(new FingerObject(_fingerPrefab, 7, 8));

            leftHand.Add(new FingerObject(_fingerPrefab, 9, 10));
            leftHand.Add(new FingerObject(_fingerPrefab, 11, 12));

            leftHand.Add(new FingerObject(_fingerPrefab, 13, 14));
            leftHand.Add(new FingerObject(_fingerPrefab, 15, 16));

            leftHand.Add(new FingerObject(_fingerPrefab, 17, 18));
            leftHand.Add(new FingerObject(_fingerPrefab, 19, 20));

            _handObjects.Add(leftHand);

            List<HandObject> rightHand = new();
            rightHand.Add(new PalmObject(_palmPrefab, 0, 5, 17));

            rightHand.Add(new FingerObject(_fingerPrefab, 1, 2));
            rightHand.Add(new FingerObject(_fingerPrefab, 3, 4));

            rightHand.Add(new FingerObject(_fingerPrefab, 5, 6));
            rightHand.Add(new FingerObject(_fingerPrefab, 7, 8));

            rightHand.Add(new FingerObject(_fingerPrefab, 9, 10));
            rightHand.Add(new FingerObject(_fingerPrefab, 11, 12));

            rightHand.Add(new FingerObject(_fingerPrefab, 13, 14));
            rightHand.Add(new FingerObject(_fingerPrefab, 15, 16));

            rightHand.Add(new FingerObject(_fingerPrefab, 17, 18));
            rightHand.Add(new FingerObject(_fingerPrefab, 19, 20));

            _handObjects.Add(rightHand);
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
            float ar = (float)image.Width() / image.Height();

            lock (_landmarkLock)
            {
                _handLandmarks = new List<List<Vector3>>();
                _handIsRight = new List<bool>();
                for (int i = 0; i < handCount; ++i)
                {
                    _handLandmarks.Add(new List<Vector3>());

                    string categoryName = result.handedness[i].categories[0].categoryName;
                    _handIsRight.Add(categoryName == "Right");
                    var rootLandmark2D = result.handLandmarks[i].landmarks[0];
                    Vector3 handOffset = new Vector2((rootLandmark2D.x - 0.5f) * ar, 1f - rootLandmark2D.y);

                    for (int j = 0; j < result.handWorldLandmarks[i].landmarks.Count; ++j)
                    {
                        var landmark = result.handWorldLandmarks[i].landmarks[j];
                        Vector3 pos = TransformPoint(new Vector3(landmark.x, landmark.y, landmark.z), handOffset);
                        _handLandmarks[i].Add(pos);
                    }
                }
                _isStale = true;
            }
        }


        private Vector3 TransformPoint(Vector3 landmark, Vector3 offset)
        {
            landmark *= _spacing;
            if (_flipY)
            {
                landmark = new Vector3(landmark.x, -landmark.y, landmark.z);
            }

            if(!_mirror) // data is mirrored by default
            {
                landmark = new Vector3(landmark.x, landmark.y, -landmark.z);
            }

            landmark += offset * _handOffsetMagnitude;

            return landmark;
        }


        private void LateUpdate()
        {
            if (_isStale)
            {
                RefreshHandObjects();
                _lastUpdateTime = Time.time;
                _objectsHidden = false;
                _isStale = false;
            }
            else if (Time.time > _lastUpdateTime + 0.5f)
            {
                HideHandObjects(0); // for now only 2 hands
                HideHandObjects(1);
                _lastUpdateTime = Time.time + 1000f;
                _objectsHidden = true;
            }
        }


        private void RefreshHandObjects()
        {
            lock (_landmarkLock)
            {
                int handCount = _handObjects.Count;
                for (int i = 0; i < handCount; ++i)
                {
                    if(i < _handLandmarks.Count)
                    {
                        foreach (HandObject obj in _handObjects[i])
                        {
                            obj.Update(_handLandmarks[i]);
                        }
                    }
                    else
                    {
                        HideHandObjects(i);
                    }
                }
            }
        }


        private void HideHandObjects(int handIndex)
        {
            if(handIndex < _handObjects.Count)
            {
                foreach (var obj in _handObjects[handIndex])
                {
                    obj.Hide();
                }
            }
        }


        private abstract class HandObject
        {
            public virtual void Update(List<Vector3> handLandmarks) { }
            public virtual void Hide() { }
        }


        private class PalmObject : HandObject
        {
            GameObject _gameObject;
            private int _rootIndex;
            private int _innerIndex;
            private int _outerIndex;

            public PalmObject(GameObject prefab, int rootIndex, int innerIndex, int outerIndex)
            {
                _gameObject = Instantiate(prefab);
                _rootIndex = rootIndex;
                _innerIndex = innerIndex;
                _outerIndex = outerIndex;
            }

            public override void Update(List<Vector3> handLandmarks)
            {
                _gameObject.transform.position = handLandmarks[_rootIndex];

                Vector3 a = handLandmarks[_rootIndex];
                Vector3 b = handLandmarks[_innerIndex];
                Vector3 c = handLandmarks[_outerIndex];

                Vector3 normal = Vector3.Normalize(Vector3.Cross(b - a, c - a));
                Vector3 up = ((b + c) * 0.5f - a).normalized;
                _gameObject.transform.rotation = Quaternion.LookRotation(normal, up);
                _gameObject.SetActive(true);
            }

            public override void Hide()
            {
                _gameObject.SetActive(false);
            }
        }


        private class FingerObject : HandObject
        {
            GameObject _gameObject;
            private int _baseIndex;
            private int _tipIndex;

            public FingerObject(GameObject prefab, int baseIndex, int tipIndex)
            {
                _gameObject = Instantiate(prefab);
                _baseIndex = baseIndex;
                _tipIndex = tipIndex;
            }

            public override void Update(List<Vector3> handLandmarks)
            {
                Vector3 basePos = handLandmarks[_baseIndex];
                Vector3 tipPos = handLandmarks[_tipIndex];
                _gameObject.transform.position = tipPos;
                _gameObject.transform.rotation = Quaternion.LookRotation((basePos - tipPos).normalized);
                _gameObject.SetActive(true);
            }

            public override void Hide()
            {
                _gameObject.SetActive(false);
            }
        }


        private class NodeObject : HandObject
        {
            GameObject _gameObject;
            private int _index;

            public NodeObject(GameObject prefab, int index)
            {
                _gameObject = Instantiate(prefab);
                _index = index;
            }

            public override void Update(List<Vector3> handLandmarks)
            {
                _gameObject.transform.position = handLandmarks[_index];
                _gameObject.SetActive(true);
            }

            public override void Hide()
            {
                _gameObject.SetActive(false);
            }
        }


#if UNITY_EDITOR
        private void OnGUI()
        {
            if (_objectsHidden) return;

            //lock (_landmarkLock)
            //{
            //    for (int i = 0; i < _handLandmarks.Count; ++i)
            //    {
            //        for(int j = 0; j < _handLandmarks[i].Count; ++j)
            //        {
            //            Handles.Label(_handLandmarks[i][j], j.ToString());
            //        }
            //    }
            //}
        }
#endif
    }
}
