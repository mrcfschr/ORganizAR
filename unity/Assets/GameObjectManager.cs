using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class GameObjectManager : MonoBehaviour
{
    public GameObject remote;
    private RemoteUnityScene remoteScene;

    void Start()
    {
        // Initialize and get reference to RemoteUnityScene script
        remoteScene = remote.GetComponent<RemoteUnityScene>();
        if (remoteScene == null)
        {
            Debug.LogError("RemoteUnityScene component is not found on the object.");
            return;
        }

        // Simulate creating a sphere
        uint sphereKey = CreateSphere();
        // Set the sphere visible after a delay
        StartCoroutine(InitializeSphere(sphereKey, 3f)); // Delay of 3 seconds
    }

    uint CreateSphere()
    {
        // Define the message for creating a sphere (assuming '0' is the command for creating a sphere)
        byte[] createSphereData = BitConverter.GetBytes((uint)PrimitiveType.Sphere);
        uint key = remoteScene.ProcessMessage(0, createSphereData);
        Debug.Log($"Sphere created with key: {key}");
        return key;
    }

    System.Collections.IEnumerator InitializeSphere(uint key, float delay)
    {
        yield return new WaitForSeconds(delay);

        // Set the sphere visible
        SetActive(key, true);

        // Set the local transform of the sphere
        SetLocalTransform(key);
    }

    void SetActive(uint key, bool isActive)
    {
        byte[] setActiveData = new byte[8];
        Array.Copy(BitConverter.GetBytes(key), setActiveData, 4);
        Array.Copy(BitConverter.GetBytes(isActive ? 1 : 0), 0, setActiveData, 4, 4);
        uint result = remoteScene.ProcessMessage(1, setActiveData);
        Debug.Log($"Set sphere visible result: {result}");
    }

    void SetLocalTransform(uint key)
    {
        Vector3 position = new Vector3(0.2f, 0.5f, 1f);
        Quaternion rotation = Quaternion.identity;
        Vector3 scale = new Vector3(0.001f, 0.001f, 0.001f);

        // Create byte array for position, rotation, and scale
        byte[] transformData = new byte[44];
        Array.Copy(BitConverter.GetBytes(key), transformData, 4);
        Array.Copy(BitConverter.GetBytes(position.x), 0, transformData, 4, 4);
        Array.Copy(BitConverter.GetBytes(position.y), 0, transformData, 8, 4);
        Array.Copy(BitConverter.GetBytes(position.z), 0, transformData, 12, 4);
        Array.Copy(BitConverter.GetBytes(rotation.x), 0, transformData, 16, 4);
        Array.Copy(BitConverter.GetBytes(rotation.y), 0, transformData, 20, 4);
        Array.Copy(BitConverter.GetBytes(rotation.z), 0, transformData, 24, 4);
        Array.Copy(BitConverter.GetBytes(rotation.w), 0, transformData, 28, 4);
        Array.Copy(BitConverter.GetBytes(scale.x), 0, transformData, 32, 4);
        Array.Copy(BitConverter.GetBytes(scale.y), 0, transformData, 36, 4);
        Array.Copy(BitConverter.GetBytes(scale.z), 0, transformData, 40, 4);

        uint result = remoteScene.ProcessMessage(2, transformData); // Assuming '6' is for setting local transform
        Debug.Log($"Set local transform result: {result}");
    }

}