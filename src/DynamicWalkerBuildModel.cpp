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

		// Connect the platform to the ground
		Vec3 locationInParent(0.0, 0.0, 0.0);
		Vec3 orientationInParent(0.0, 0.0, 0.0);
		Vec3 locationInChild(0.0, 0.0, 0.0);
		Vec3 orientationInChild(0.0, 0.0, 0.0);
		PinJoint *platformToGround = new PinJoint("PlatformToGround", ground, locationInParent, orientationInParent, *platform, locationInChild, orientationInChild, false);

		CoordinateSet &platformJoints = platformToGround->upd_CoordinateSet();
		platformJoints[0].setName("platform_rz");
		double rotRangePlatform[2] = {-Pi / 2.0, 0};
		platformJoints[0].setRange(rotRangePlatform);
		platformJoints[0].setDefaultValue(convertDegreesToRadians(-10.0));
		platformJoints[0].setDefaultLocked(true);

		// Pelvis
		double pelvisMass = 1;
		Vec3 pelvisCoM(0.0, 0.0, 0.0);
		Inertia pelvisInertia(1.0, 1.0, 1.0, 0.0, 0.0, 0.0);

		OpenSim::Body* pelvis = new OpenSim::Body("Pelvis", pelvisMass, pelvisCoM, pelvisInertia);

		locationInParent = Vec3(0.0, 0.0, 0.0);
		orientationInParent = Vec3(0.0, 0.0, 0.0);
		locationInChild = Vec3(0.0, 0.0, 0.0);
		orientationInChild = Vec3(0.0, 0.0, 0.0);

		FreeJoint *pelvisToPlatform = new FreeJoint("PelvisToPlatform", *platform,
				locationInParent, orientationInParent, *pelvis, locationInChild,
				orientationInChild, false);

		CoordinateSet &pelvisJointCoords = pelvisToPlatform->upd_CoordinateSet();

		pelvis->addDisplayGeometry("sphere.vtp");
		pelvis->updDisplayer()->setScaleFactors(Vec3(pelvisWidth / 2.0, pelvisWidth / 2.0, pelvisWidth));

		osimModel.addBody(pelvis);

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
