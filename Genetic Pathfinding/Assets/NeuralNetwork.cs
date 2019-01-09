using System;
using System.Collections.Generic;
//using System.Diagnostics;
using UnityEngine;

public class NeuralNetwork : IComparable<NeuralNetwork>{
  private int[] layers;
  float[][] neurons;
  public bool copied;
  public int type;
  public float[][][] weights;
  private float bias = 0.2f;
  private float fittness;
  public int age;
  
  public NeuralNetwork(int[] layers)
  {
    age = 1;
    copied = false;
    type = 0;
    this.layers = new int[layers.Length];

    for (int i = 0; i < layers.Length; i++)
    {
      this.layers[i] = layers[i];
    }

    InitNeurons();
    weights = InitWeights();
  }
  
  private double sigmoid(double x)
  {
      //assert((1 + Math.Exp(-x)) != 0);
      return 1 / (1 + Math.Exp(-x));
  }

  private void InitNeurons()
  {
    //List to jagged array
    List<float[]> neuronsList = new List<float[]>();

    for (int i = 0; i < layers.Length; i++)
    {
      neuronsList.Add(new float[layers[i]]);
    }

    neurons = neuronsList.ToArray();
  }

  public NeuralNetwork(NeuralNetwork copyNetwork)
  {
    age = copyNetwork.age + 1;
    this.layers = new int[copyNetwork.layers.Length];
    for (int i = 0; i < copyNetwork.layers.Length; i++)
    {
      this.layers[i] = copyNetwork.layers[i];
    }
    copied = true;
    type = copyNetwork.type;

    InitNeurons();

    weights = InitWeights();
    CopyWeights(copyNetwork.weights);

  }

  private void CopyWeights(float[][][] weights)
  {
    for (int i = 0; i < weights.Length; i++)
    {
      for (int j = 0; j < weights[i].Length; j++)
      {
        for (int k = 0; k < weights[i][j].Length; k++)
        {
          this.weights[i][j][k] = weights[i][j][k];
        }
      }
    }
  }

  private float[][][] InitWeights()
  {
    List<float[][]> weightsList = new List<float[][]>();

    for (int i = 1; i < layers.Length; i++)
    {
      List<float[]> layerWeightList = new List<float[]>();

      int neuronsInPrevLayer = layers[i - 1];
      for (int j = 0; j < neurons[i].Length; j++) {
        float[] neuronWeights = new float[neuronsInPrevLayer];

        for (int k = 0; k < neuronsInPrevLayer; k++)
        {
          neuronWeights[k] = UnityEngine.Random.Range(-0.5f, 0.5f);
        }

        layerWeightList.Add(neuronWeights);
      }

      weightsList.Add(layerWeightList.ToArray());
    }

    return weightsList.ToArray();
  }

  public float[] FeedForward(float[] inputs)
  {
    for (int i = 0; i < inputs.Length; i++)
    {
      neurons[0][i] = inputs[i];
    }

    for (int i = 1; i < layers.Length; i++)
    {
      //Iterate over every layer
      for (int j = 0; j < neurons[i].Length; j++)
      {
        //Iterate over every neuron in layer

        float summation = bias;

        for (int k = 0; k < neurons[i - 1].Length; k++)
        {
          //Iterate over every neuron in previous layer
          
          summation += weights[i - 1][j][k] * neurons[i - 1][k];
        }

        neurons[i][j] = (float)Math.Tanh(summation);
      }
    }

    // return output layer
    return neurons[neurons.Length-1];
  }

  public NeuralNetwork Breed(NeuralNetwork otherNet)
    {
		NeuralNetwork nn = new NeuralNetwork (otherNet);
		nn.type = 2;
    for (int i = 0; i < this.weights.Length; i++)
    {
      for (int j = 0; j < this.weights[i].Length; j++)
      {
        for (int k = 0; k < this.weights[i][j].Length; k++)
        {
          if (UnityEngine.Random.Range(0, 1) >= 0.5) {
						nn.weights[i][j][k] =  (float) Math.Tanh(otherNet.weights[i][j][k]);
          } else
          {
						nn.weights[i][j][k] =  (float) Math.Tanh(this.weights[i][j][k]);
          }
        }
      }
    }
		return nn.Mutate (Manager.mutatePercent);
  }

	public NeuralNetwork Mutate(int mutation)
  {
    NeuralNetwork nn = new NeuralNetwork(layers);
    //float[][][] weights = InitWeights();
    nn.type = 1;
    //nn.copied = true;

    for (int i = 0; i < weights.Length; i++)
    {
      for (int j = 0; j < weights[i].Length; j++)
      {
        for (int k = 0; k < weights[i][j].Length; k++)
        {
          float weight = weights[i][j][k];
          
          // Mutate Weight
          float randomNumber = UnityEngine.Random.Range(0, 1000);

					if (randomNumber <= 2f * mutation)
          {
            weight *= -1;
          }
					else if (randomNumber <= 4f * mutation)
          {
            weight = UnityEngine.Random.Range(-0.5f, 0.5f);
          }
					else if (randomNumber <= 10f * mutation)
          {
						float factor = UnityEngine.Random.Range (0f, 2f) - 1;
            weight *= factor;

          }
					if (randomNumber <= 10f * mutation)
						nn.weights [i] [j] [k] = (float)Math.Tanh (weight);
					else
						nn.weights [i] [j] [k] = weight;
        }
      }
    }
    return nn;
  }

  public void AddFitness(float fit)
  {
    fittness += fit;
  }

  public void SetFitness(float fit)
  {
    fittness = fit;
  }

  public float GetFitness()
  {
    return fittness;
  }

  public int CompareTo(NeuralNetwork other)
  {
    if (other == null) return 1;

    if (fittness > other.fittness)
      return 1;
    if (fittness < other.fittness)
      return -1;

    return 0;
  }
}
