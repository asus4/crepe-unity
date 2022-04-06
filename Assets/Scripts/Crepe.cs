namespace TensorFlowLite
{
    using UnityEngine;

    public sealed class Crepe : System.IDisposable
    {
        private readonly Interpreter interpreter;

        private readonly float[] input;
        private readonly float[] output;

        public Crepe(string modelPath, bool useGPU)
        {
            var options = new InterpreterOptions();
            if (useGPU)
            {
                options.AddGpuDelegate();
            }
            else
            {
                options.threads = SystemInfo.processorCount;
            }

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
            interpreter.ResizeInputTensor(0, inputShape0);
            interpreter.AllocateTensors();
        }

        public void Dispose()
        {
            interpreter?.Dispose();
        }

        public float[] Predict(float[] input)
        {
            interpreter.SetInputTensorData(0, input);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, output);
            return output;
        }
    }
}
