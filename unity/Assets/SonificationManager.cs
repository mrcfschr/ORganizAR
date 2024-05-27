using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SonificationManager : MonoBehaviour
{
    public List<GameObject> allQuads = new List<GameObject>();
    public List<GameObject> randomQuads = new List<GameObject>();
    public AudioSource preConfiguredSource;
    private bool isInitialized = false;

    public void Awake()
    {
        preConfiguredSource = Resources.Load<AudioSource>("SpatialAudioSource");
    }

    public void AddAudioToRandomQuads(int count)
    {
        for (int i = 0; i < count; i++)
        {
            int index = Random.Range(0, allQuads.Count);
            if (!randomQuads.Contains(allQuads[index]))
            {
                randomQuads.Add(allQuads[index]);
                Instantiate(preConfiguredSource, allQuads[index].transform);
                allQuads[index].AddComponent<ActivateAudio>();
            }
            else
            {
                i--;
            }
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (!isInitialized && GameObject.FindObjectsOfType<GameObject>().Length > 100)
        {
            GameObject[] allObjects = GameObject.FindObjectsOfType<GameObject>();
            foreach (GameObject obj in allObjects)
            {
                if (obj.GetComponent<MeshFilter>() != null && obj.GetComponent<MeshRenderer>() != null)
                {
                    allQuads.Add(obj);
                }
            }
            isInitialized = true;
            AddAudioToRandomQuads(10);
        }
    }
}
