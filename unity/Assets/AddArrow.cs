using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AddArrow : MonoBehaviour
{
    public Transform targetTransform; // Reference to the transform to align the arrow with

    void Start()
    {
    
        GameObject arrowPrefab = Resources.Load<GameObject>("3D RightArrow");
        targetTransform = this.transform;

        if (arrowPrefab != null)
        {

            GameObject arrowInstance = Instantiate(arrowPrefab, Vector3.zero, Quaternion.identity);

            Transform arrowTip = arrowInstance.transform.Find("Sphere");

            if (arrowTip != null)
            {
                Vector3 directionToTarget = targetTransform.position - arrowTip.position;

                arrowInstance.transform.position = targetTransform.position - directionToTarget;

                arrowInstance.transform.rotation = Quaternion.FromToRotation(Vector3.right, Vector3.down);

                arrowInstance.transform.rotation *= Quaternion.FromToRotation(arrowTip.localPosition, directionToTarget);
            }
            else
            {
                Debug.LogError("Arrow tip (Sphere) not found in the prefab.");
            }
        }
        else
        {
            Debug.LogError("Arrow prefab not found in Resources folder.");
        }
    }
}
