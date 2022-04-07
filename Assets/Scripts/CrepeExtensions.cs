namespace TensorFlowLite
{
    using System;
    using UnityEngine.Assertions;
    using Unity.Collections;

    public static class CrepeExtensions
    {
        public static float Sum(NativeArray<float> arr)
        {
            double sum = 0;
            for (int i = 0; i < arr.Length; i++)
            {
                sum += arr[i];
            }
            return (float)sum;
        }

        public static float Mean(this float[] arr)
        {
            double sum = 0;
            for (int i = 0; i < arr.Length; i++)
            {
                sum += arr[i];
            }
            return (float)(sum / arr.Length);
        }

        public static float StdDev(this float[] arr)
        {
            double mean = arr.Mean();
            double sum = 0;
            for (int i = 0; i < arr.Length; i++)
            {
                sum += (arr[i] - mean) * (arr[i] - mean);
            }
            return (float)Math.Sqrt(sum / arr.Length);
        }

        public static float[] Add(this float[] arr, float n)
        {
            for (int i = 0; i < arr.Length; i++)
            {
                arr[i] += n;
            }
            return arr;
        }

        public static float[] Subtract(this float[] arr, float n)
        {
            for (int i = 0; i < arr.Length; i++)
            {
                arr[i] -= n;
            }
            return arr;
        }

        public static NativeArray<float> Multiply(this NativeSlice<float> a, NativeSlice<float> b, Allocator allocator)
        {
            Assert.AreEqual(a.Length, b.Length);
            var arr = new NativeArray<float>(a.Length, allocator);
            for (int i = 0; i < a.Length; i++)
            {
                arr[i] = a[i] * b[i];
            }
            return arr;
        }

        public static float[] Divide(this float[] arr, float n)
        {
            for (int i = 0; i < arr.Length; i++)
            {
                arr[i] /= n;
            }
            return arr;
        }

        public static (int index, float value) ArgMax(this NativeArray<float> arr)
        {
            int index = 0;
            float value = arr[0];
            for (int i = 1; i < arr.Length; i++)
            {
                if (arr[i] > value)
                {
                    index = i;
                    value = arr[i];
                }
            }
            return (index, value);
        }

        public static float[] LinerSpace(float start, float end, int num)
        {
            float[] arr = new float[num];
            float d = (end - start) / num;
            for (int i = 0; i < num; i++)
            {
                arr[i] = start + d * i;
            }
            return arr;
        }


    }
}
