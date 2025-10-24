using UnityEngine;

namespace Mediapipe.Unity.Sample.HandLandmarkDetection
{
    public class HandPart : MonoBehaviour
    {
        HandPartType m_Type;
        HandPart m_TouchedHandPart;


        public HandPartType Type { get { return m_Type; } set { m_Type = value; } }

        private void OnTriggerEnter(Collider other)
        {
            if (m_TouchedHandPart != null || gameObject.activeInHierarchy == false) return;

            if (other.TryGetComponent(out HandPart otherHandPart))
            {
                if (m_Type != HandPartType.Thumb && otherHandPart.Type == HandPartType.Thumb)
                {
                    m_TouchedHandPart = otherHandPart;
                    GameManager.Instance.HandPartsInContact(this, otherHandPart);
                }
            }
        }


        private void OnTriggerExit(Collider other)
        {
            if (other.TryGetComponent(out HandPart otherHandPart))
            {
                if (m_TouchedHandPart == otherHandPart)
                {
                    m_TouchedHandPart = null;
                    GameManager.Instance.HandPartsOutOfContact(this, otherHandPart);
                }
            }
        }
    }

    public enum HandPartType
    {
        None,

        Palm,
        Thumb,
        IndexFinger,
        MiddleFinger,
        RingFinger,
        PinkyFinger
    }
}
