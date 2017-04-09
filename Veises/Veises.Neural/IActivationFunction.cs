namespace Veises.Neural
{
	public interface IActivationFunction
	{
		double Activate(double sum);

		double Deactivate(double output);
	}
}