namespace Veises.Neural
{
	public interface IErrorFunction
	{
		double Calculate(double output, double target);
	}
}