using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ActivateAudio : MonoBehaviour
{
    private AudioSource audioSource;
    private MeshRenderer meshRenderer;
    void Start()
    {

        audioSource = GetComponentInChildren<AudioSource>();
        meshRenderer = GetComponent<MeshRenderer>();
        audioSource.loop = true;
    }
    void Update()
    {
        
        if (gameObject.activeInHierarchy && meshRenderer.enabled)
        {
            if (!audioSource.isPlaying)
            {
                audioSource.Play(); 
            }
        }
        else
        {
            if (audioSource.isPlaying)
            {
                audioSource.Stop(); 
            }
        }
    }
}
