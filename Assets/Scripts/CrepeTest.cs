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
    private bool useGPU = false;
    [SerializeField]
    private AudioClip audioClip;

    private Crepe crepe;
    private float[] inputSamples = new float[1024];
    private int index = 0;
    private StringBuilder sb = new StringBuilder();

    private void Start()
    {
        crepe = new Crepe(modelPath, useGPU);
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
            float[] output = crepe.Predict(inputSamples);

            sb.Clear();
            sb.Append($"[{index}]=");
            for (int i = 0; i < output.Length; i++)
            {
                sb.Append($"{output[i]:F4}  ");
            }
            Debug.Log(sb.ToString());

            index += 1024;
        }
    }
}
