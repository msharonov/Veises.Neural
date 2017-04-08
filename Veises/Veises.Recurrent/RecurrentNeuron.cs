using Veises.Neural;

namespace Veises.Recurrent
{
	public sealed class RecurrentNeuron: Neuron
	{
		private readonly Axon _contextAxon;

		private readonly Neuron _contextNeuron;

		public RecurrentNeuron(IActivationFunction activationFunction, Bias bias)
			: base(activationFunction, bias)
		{
			_contextNeuron = new Neuron(activationFunction, bias);

			_contextAxon = new Axon(_contextNeuron, this);
		}

		public override void CalculateOutput()
		{
			var inputSum = 0d;

			foreach (var axon in _inputAxons)
			{
				inputSum += axon.GetOutput();
			}

			inputSum += _bias.Weight;

			inputSum += _contextAxon.GetOutput();

			Output = _activationFunction.Activate(inputSum);

			_contextNeuron.SetInput(Output);
		}
	}
}