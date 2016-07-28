#include <iostream>
#include <string>
#include <QApplication.h>
#include <QVTKWidget.h>

#include "uMath.h"
#include "utils.h"
#include "itk_io.h"
#include "MatioWrapper.h"
#include "ITKImportWrapper.h"
#include "RenderWindowUISingleInheritance.h"

#include "./cudaKernels.cuh"

#include "itkImage.h"
#include "itkTranslationTransform.h"
#include "itkEuler3DTransform.h"
#include "itkImageFileReader.h"
#include "itkNormalizeImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkSliceBySliceImageFilter.h"
#include "itkForwardFFTImageFilter.h"
#include "itkInverseFFTImageFilter.h"
#include "itkFFTWForwardFFTImageFilter.h"
#include "itkFFTWInverseFFTImageFilter.h"
#include "itkFFTShiftImageFilter.h"
#include "itkCyclicShiftImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkAffineTransform.h"
#include "itkAddImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkDivideImageFilter.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkImportImageFilter.h"
#include "itkExtractImageFilter.h"
#include "itkComplexToRealImageFilter.h"
#include "itkComplexToImaginaryImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkStatisticsImageFilter.h"
#include "itkAbsImageFilter.h"
#include "itkImageDuplicator.h"
#include "itkTileImageFilter.h"
#include "itkImageToVTKImageFilter.h"

#include <vtkVersion.h>
#include <vtkSmartPointer.h>
#include <vtkSphere.h>
#include <vtkSampleFunction.h>
#include <vtkSmartVolumeMapper.h>
#include <vtkColorTransferFunction.h>
#include <vtkPiecewiseFunction.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkVolumeProperty.h>
#include <vtkCamera.h>
#include <vtkImageShiftScale.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkXMLImageDataReader.h>
#include <vtkCommand.h>

typedef float                                 SimPixelType;
typedef itk::Image< SimPixelType, 3 > SimImageType;
typedef itk::ImportImageFilter< SimPixelType, 3 >   ImportFilterType;
// ITK to VTK image converter
typedef itk::ImageToVTKImageFilter<SimImageType> VTKConverterType;
char sig = ' ';
bool use_saved_instance = 0;

using namespace std;



int main(int argc, char *argv[])
{
	const unsigned int nx = 128;
	const unsigned int ny = 128;
	const unsigned int nz = 128;
	float theta = 0;
	float phi = 0;
	QApplication app( argc, argv );
	vtkSmartPointer<vtkRenderer> ren1 = vtkSmartPointer<vtkRenderer>::New();
	RenderWindowUISingleInheritance renderWindowUISingleInheritance(ren1);
	renderWindowUISingleInheritance.show();
	ren1->SetBackground(0.5,0.5,0.5);
	renderWindowUISingleInheritance.dumpLog("Welcome", 1, 0);
	float* cube = new float[ nx * ny * nz ];
	float* output = new float[ nx * ny * nz ];		
	float* dev_in;
	float* dev_out;

	cuRotateInit(
		&dev_in,
		&dev_out,
		nx, ny, nz
		);
	
	vtkSmartPointer<vtkVolumeProperty> volumeProperty = vtkSmartPointer<vtkVolumeProperty>::New();
	volumeProperty->ShadeOff();
	volumeProperty->SetInterpolationType(VTK_LINEAR_INTERPOLATION);
	vtkSmartPointer<vtkPiecewiseFunction> compositeOpacity = vtkSmartPointer<vtkPiecewiseFunction>::New();
	compositeOpacity->AddPoint(0.0,0.0);
	compositeOpacity->AddPoint(0.5,0.5);
	compositeOpacity->AddPoint(1.0,1.0);
	volumeProperty->SetScalarOpacity(compositeOpacity); // composite first.
	vtkSmartPointer<vtkColorTransferFunction> color = vtkSmartPointer<vtkColorTransferFunction>::New();
	color->AddRGBPoint(0.0  ,0.0,0.0,0.0);
	color->AddRGBPoint(1.0, 0.0,1.0,1.0);
	volumeProperty->SetColor(color);
	vtkSmartPointer<vtkVolume> volume = vtkSmartPointer<vtkVolume>::New();
	SimImageType::Pointer cube_image;
	vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
	VTKConverterType::Pointer converter = VTKConverterType::New();
	vtkSmartPointer<vtkSmartVolumeMapper> volumeMapper = vtkSmartPointer<vtkSmartVolumeMapper>::New();
	bool continueFlag = true;
#ifdef WIN32
	while(1) {
		// Program exiting logic
		while (sig != 's') {
			app.processEvents();
			Sleep(30);
			if (sig == 'v')
			{
				continueFlag = false;
				break;
			}
		}
		if (!continueFlag) {
			break;
		}
		sig = ' ';
#endif
		phi = renderWindowUISingleInheritance.getPhi();
		theta = renderWindowUISingleInheritance.getTheta();
		renderWindowUISingleInheritance.dumpLogStr(
			std::string("phi: ") + std::to_string(phi) + std::string(", theta: ") + std::to_string(theta),
			0,
			1
			);


		for (unsigned int i = 0; i < nx * ny * nz; i++) {
			output[i] = 0.0;
		}

		std::string filename = renderWindowUISingleInheritance.getMatFilename().toStdString();
		std::string varname = renderWindowUISingleInheritance.getMatVarName().toStdString();
		MatioWrapper* matio_instance = new MatioWrapper( filename.c_str() );
		matio_instance->Open();
		matio_instance->Read<double>(varname.c_str(), cube, nx * ny * nz);
		matio_instance->~MatioWrapper();

		cuRotate(
			&cube,
			&output,
			&dev_in,
			&dev_out,
			nx, ny, nz,
			theta,
			phi
			);
		
		const bool importImageFilterWillOwnTheBuffer = false;
		ITKImportWrapper<float, 3> itkImporter(output, nx, ny, nz);

		cube_image = itkImporter.returnITKImage();
		converter->SetInput(cube_image);
		try {
			converter->Update();
		}
		catch( itk::ExceptionObject & error ) {
			std::cerr << "Error: " << error << std::endl;
			return EXIT_FAILURE;
		}
		imageData = converter->GetOutput();
	    volumeMapper->SetBlendModeToComposite(); // composite first
#if VTK_MAJOR_VERSION <= 5
		volumeMapper->SetInputConnection(imageData->GetProducerPort());
#else

	    volumeMapper->SetInputData(imageData);
#endif
	    

	    volume->SetMapper(volumeMapper);
	    volume->SetProperty(volumeProperty);
		volume->Update();
	    ren1->AddViewProp(volume);
	    ren1->ResetCamera();
	    renderWindowUISingleInheritance.updateQVTKWidget();
		app.processEvents();
		
		// itkImporter.~ITKImportWrapper();
		/*std::string output_filename( "D:/RLSimulation/itk_projects/testMatGUI/build/output/output.mha" );
		itk_io::store_mha(
			output,
			3,
			nx,
			ny,
			nz,
			output_filename.c_str()
			);*/
	}

    std::string output_filename( "D:/RLSimulation/itk_projects/testMatGUI/build/output/output.mha" );
    itk_io::store_mha(
    	output,
    	3,
    	nx,
    	ny,
    	nz,
    	output_filename.c_str()
    	);
		
	cuRotateFree(
		&dev_in,
		&dev_out
		);
	delete cube;
	delete output;
	return 0;
}