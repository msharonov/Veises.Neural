using System;
using System.Collections.Generic;

namespace Veises.Recurrent
{
	public sealed class Neuron
	{
		private readonly IList<Axon> _outputAxons;

		private readonly IList<Axon> _inputAxons;

		public double Error { get; private set; }

		public double Output { get; private set; }

		public Neuron()
		{
			_outputAxons = new List<Axon>();
			_inputAxons = new List<Axon>();
		}

		public void AddChild(Axon axon)
		{
			if (axon == null)
				throw new ArgumentNullException(nameof(axon));

			_outputAxons.Add(axon);
		}

		public void AddParent(Axon axon)
		{
			if (axon == null)
				throw new ArgumentNullException(nameof(axon));

			_inputAxons.Add(axon);
		}

		public double CalculateOutput()
		{
			var input = 0d;

			foreach (var axon in _inputAxons)
			{
				input += axon.WeightedOutput;
			}

			return Output = 1d / (1 + Math.Exp(-input));
		}

		public void SetOutput(double output)
		{
			if (_inputAxons.Count > 0)
				throw new ArgumentException(nameof(output));

			Output = output;
		}

		public void AdjustWeights()
		{
			foreach (var axon in _outputAxons)
			{
				axon.AdjustWeight();
			}
		}
	}
}