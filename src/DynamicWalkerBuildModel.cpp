#include <OpenSim/OpenSim.h>

using namespace OpenSim;
using namespace SimTK;

int main()
{
	try {
		// Constants
		double g = 9.81, pelvisWidth = 0.20, thighLength = 0.40, shankLength = 0.435;

		// Initialize the model
		Model osimModel = Model();
		osimModel.setName("DynamicWalkerModel");

		// Ground
		OpenSim::Body &ground = osimModel.getGroundBody();

		// Acceleratio due to gravity
		osimModel.setGravity(Vec3(0, -g, 0));

		// Platform
		double platformMass = 1;
		Vec3 platformCoM(0.0, 0.0, 0.0); // Center of mass wrt local origin
		Inertia platformInertia(1.0, 1.0, 1.0, 0.0, 0.0, 0.0);

		OpenSim::Body* platform = new OpenSim::Body("Platform", platformMass, platformCoM, platformInertia);

		platform->addDisplayGeometry("box.vtp");
		platform->updDisplayer()->setScaleFactors(Vec3(1, 0.05, 1));

		osimModel.addBody(platform);

		// Print XML version of the model
		osimModel.print("DynamicWalkerModel.osim");
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
