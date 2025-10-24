using System.Collections.Generic;
using UnityEngine;


namespace Mediapipe.Unity.Sample.HandLandmarkDetection
{
    public class GameManager : MonoBehaviour
    {
        public static GameManager Instance { get; private set; }

        [SerializeField] UnityEngine.Color m_DefaultColor = UnityEngine.Color.white;
        [SerializeField] List<HandPartColor> m_HandPartColors = new();

        HandPart m_DictatingHandPart;


        void Awake()
        {
            Instance = this;
        }


        void Update()
        {

        }


        public void HandPartsInContact(HandPart a, HandPart b)
        {
            if (m_DictatingHandPart == null)
            {
                m_DictatingHandPart = a;
                foreach (var handPartColor in m_HandPartColors)
                {
                    if (m_DictatingHandPart.Type == handPartColor.Type)
                    {
                        Debug.Log($"Selected color {handPartColor.Color}");
                        break;
                    }
                }
            }
        }

        public void HandPartsOutOfContact(HandPart a, HandPart b)
        {
            if (m_DictatingHandPart == a)
            {
                m_DictatingHandPart = null;
                Debug.Log($"Selected color None");
            }
        }

        [System.Serializable]
        struct HandPartColor
        {
            public HandPartType Type;
            public UnityEngine.Color Color;
        }
    }
}