using System;
using UnityEngine;
using UnityEngine.Assertions;
using TensorFlowLite;
using System.Text;

public class CrepeTest : MonoBehaviour
{
    [SerializeField, FilePopup("*.tflite")]
    private string modelPath = "";
    [SerializeField]
    private AudioClip audioClip;

    private Crepe crepe;
    private readonly float[] inputSamples = new float[1024];
    private int index = 0;
    private StringBuilder sb = new StringBuilder();
    private float lastFreq = 0;

    private void Start()
    {
        crepe = new Crepe(modelPath);
        Assert.AreEqual(1, audioClip.channels);
    }

    private void OnDestroy()
    {
        crepe?.Dispose();
    }

    private void Update()
    {
        if ((index + 1024) < audioClip.samples)
        {
            audioClip.GetData(inputSamples, index);
            var result = crepe.Predict(inputSamples);
            float diff = result.pitch - lastFreq;
            Debug.Log($"[{index}]={result} diff:{diff:F2}");
            index += 1024;
            lastFreq = result.pitch;
        }
    }
}
