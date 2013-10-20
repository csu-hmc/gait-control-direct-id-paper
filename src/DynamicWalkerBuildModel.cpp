#include <OpenSim/OpenSim.h>

using namespace OpenSim;
using namespace SimTK;

int main()
{
	try {
	}
	catch (OpenSim::Exception ex)
	{
		std::cout << ex.getMessage() << std::endl;
		return 1;
	}
	catch (SimTK::Exception::Base ex)
	{
		std::cout << ex.getMessage() << std::endl;
		return 1;
	}
	catch (std::exception ex)
	{
		std::cout << ex.what() << std::endl;
		return 1;
	}
	catch (...)
	{
		std::cout << "UNRECOGNIZED EXCEPTION" << std::endl;
	}
	std::cout << "OpenSim example completed sucessfully" << std::endl;
	std::cout << "Press return to continue" << std::endl;
	std::cin.get();
	return 0;
}
