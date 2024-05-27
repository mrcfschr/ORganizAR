using UnityEngine;

public class InstantiateQuads : MonoBehaviour
{
    public GameObject quadPrefab;
    public int numberOfQuads = 2000;
    public float spawnRadius = 50f;

    void Start()
    {
        for (int i = 0; i < numberOfQuads; i++)
        {
            Vector3 randomPosition = Random.insideUnitSphere * spawnRadius;
            Instantiate(quadPrefab, randomPosition, Quaternion.identity);
        }
    }
}
