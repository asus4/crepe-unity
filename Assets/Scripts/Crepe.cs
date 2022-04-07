namespace TensorFlowLite
{
    using System;
    using System.Linq;
    using UnityEngine;
    using UnityEngine.Assertions;
    using Unity.Collections;

    public sealed class Crepe : IDisposable
    {
        public readonly struct Result
        {
            public readonly float confidence;
            public readonly float pitch;

            public Result(float confidence, float pitch)
            {
                this.confidence = confidence;
                this.pitch = pitch;
            }

            public override string ToString()
            {
                return $"{Mathf.RoundToInt(pitch)}Hz : {Mathf.RoundToInt(confidence * 100)}%";
            }
        }

        private readonly Interpreter interpreter;

        private readonly float[] input;
        private readonly float[] output;
        private readonly NativeArray<float> nativeOutput;
        private readonly NativeArray<float> centsMapping;

        public Crepe(string modelPath)
        {
            var options = new InterpreterOptions()
            {
                threads = SystemInfo.processorCount,
            };
            try
            {
                interpreter = new Interpreter(FileUtil.LoadFile(modelPath), options);
            }
            catch (System.Exception e)
            {
                interpreter?.Dispose();
                throw e;
            }

            interpreter.LogIOInfo();

            int[] inputShape0 = interpreter.GetInputTensorInfo(0).shape;
            int[] outputShape0 = interpreter.GetOutputTensorInfo(0).shape;
            input = new float[inputShape0[1]];
            output = new float[outputShape0[1]];
            nativeOutput = new NativeArray<float>(outputShape0[1], Allocator.Persistent);
            Assert.AreEqual(1024, input.Length);
            Assert.AreEqual(360, output.Length);

            interpreter.ResizeInputTensor(0, inputShape0);
            interpreter.AllocateTensors();

            // initialize cents mapping
            var centsMappingArr = CrepeExtensions.LinerSpace(0, 7180, output.Length)
                .Add(1997.3794084376191f);
            centsMapping = new NativeArray<float>(centsMappingArr, Allocator.Persistent);
        }

        public void Dispose()
        {
            interpreter?.Dispose();
            nativeOutput.Dispose();
            centsMapping.Dispose();
        }

        public Result Predict(float[] audio)
        {
            PreProcess(audio);
            interpreter.SetInputTensorData(0, input);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, output);
            nativeOutput.CopyFrom(output);
            return PostProcess(nativeOutput);
        }

        private void PreProcess(float[] audio)
        {
            Assert.AreEqual(audio.Length, input.Length);
            Array.Copy(audio, input, audio.Length);

            input.Subtract((float)input.Mean());
            input.Divide((float)input.StdDev());
        }

        private Result PostProcess(NativeArray<float> activation)
        {
            (int center, float confidence) = activation.ArgMax();
            double cents = ToLocalAverageCents(activation, center);
            double frequency = 10 * Math.Pow(2, cents / 1200);
            return new Result(confidence, (float)frequency);
        }

        private double ToLocalAverageCents(NativeArray<float> activation, int center)
        {
            int start = Math.Max(0, center - 4);
            int end = Math.Min(activation.Length, center + 5);

            var salience = activation.Slice(start, end - start);
            var mapping = centsMapping.Slice(start, end - start);
            var productSum = salience.Multiply(mapping, Allocator.Temp).Sum();
            var weightSum = salience.Sum();
            return productSum / weightSum;
        }
    }
}
