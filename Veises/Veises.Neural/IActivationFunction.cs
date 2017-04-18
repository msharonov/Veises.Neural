namespace Veises.Neural
{
	public interface IActivationFunction
	{
		double Activate(double sum);

		double GetDerivative(double output);
	}
}