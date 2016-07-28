#include "RenderWindowUISingleInheritance.h"

// This is included here because it is forward declared in
// RenderWindowUISingleInheritance.h
#include "ui_RenderWindowUISingleInheritance.h"

#include <vtkPolyDataMapper.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkSphereSource.h>
#include <vtkSmartPointer.h>
#include <iostream>
// Configuration file
#include "ConfigFile.h"

extern char sig;
extern bool use_saved_instance;

// Constructor
RenderWindowUISingleInheritance::RenderWindowUISingleInheritance()
{
  this->ui = new Ui_RenderWindowUISingleInheritance;
  this->ui->setupUi(this);
 
  // Sphere
  vtkSmartPointer<vtkSphereSource> sphereSource = 
      vtkSmartPointer<vtkSphereSource>::New();
  sphereSource->Update();
  vtkSmartPointer<vtkPolyDataMapper> sphereMapper =
      vtkSmartPointer<vtkPolyDataMapper>::New();
  sphereMapper->SetInputConnection(sphereSource->GetOutputPort());
  vtkSmartPointer<vtkActor> sphereActor = 
      vtkSmartPointer<vtkActor>::New();
  sphereActor->SetMapper(sphereMapper);
 
  // VTK Renderer
  vtkSmartPointer<vtkRenderer> renderer = 
      vtkSmartPointer<vtkRenderer>::New();
  renderer->AddActor(sphereActor);
 
  // VTK/Qt wedded
  this->ui->qvtkWidget->GetRenderWindow()->AddRenderer(renderer);
 
  // Set up action signals and slots
  connect(this->ui->actionExit, SIGNAL(triggered()), this, SLOT(slotExit()));
 
}

RenderWindowUISingleInheritance::RenderWindowUISingleInheritance(vtkSmartPointer<vtkRenderer> renderer)
{
  this->ui = new Ui_RenderWindowUISingleInheritance;
  this->ui->setupUi(this);

  // VTK/Qt wedded
  this->ui->qvtkWidget->GetRenderWindow()->AddRenderer(renderer);

  mMatFilename = "";
  mMatVarName = "";
  mGeneralEdit1Text = "";
  mGeneralEdit2Text = "";
  connect(this->ui->matVarEdit, SIGNAL(textChanged(const QString &)), this, SLOT(onMatVarNameChanged()));
  connect(this->ui->matEdit, SIGNAL(textChanged(const QString &)), this, SLOT(onMatFileNameChanged()));
  connect(this->ui->generalEdit_1, SIGNAL(textChanged(const QString &)), this, SLOT(onGeneralEdit1Changed()));
  connect(this->ui->generalEdit_2, SIGNAL(textChanged(const QString &)), this, SLOT(onGeneralEdit2Changed()));
  phi = 0;
  theta = 0;
}


void RenderWindowUISingleInheritance::importCheckBoxToggled(bool toggle)
{
	if (toggle == true) {
		use_saved_instance = !toggle;
		this->dumpLog("toggle == true", 0, 1);
	} else {
		use_saved_instance = !toggle;
		this->dumpLog("toggle == false", 0, 1);
	}
}

void RenderWindowUISingleInheritance::slotExit() 
{
  sig = 'v';
  std::cout << "sig = " << sig << std::endl;
  qApp->exit();
}

void RenderWindowUISingleInheritance::closeEvent(QCloseEvent* event) 
{
  sig = 'v';
  std::cout << "sig = " << sig << std::endl;
  qApp->exit();
}

void RenderWindowUISingleInheritance::updateQVTKWidget()
{
	this->ui->qvtkWidget->update();

}

void RenderWindowUISingleInheritance::fileSelected(QString s_image, QString s_mat)
{
	mMatFilename = s_mat;
	this->ui->matEdit->setText(s_mat);
}

void RenderWindowUISingleInheritance::onMatVarNameChanged()
{
	mMatVarName = this->ui->matVarEdit->text();
}

void RenderWindowUISingleInheritance::onMatFileNameChanged()
{
	mMatFilename = this->ui->matEdit->text();
}

/*
* Slot for general line edit 1
*/
void RenderWindowUISingleInheritance::onGeneralEdit1Changed()
{
	bool ok;
    QString temp = this->ui->generalEdit_1->text();
	float temp_float = temp.toFloat(&ok);
	if (ok) {
		mGeneralEdit1Text = temp;
		phi = temp_float;
	}
}

/*
* Slot for general line edit 2
*/
void RenderWindowUISingleInheritance::onGeneralEdit2Changed()
{
	bool ok;
	QString temp = this->ui->generalEdit_2->text();
	float temp_float = temp.toFloat(&ok);
	if (ok) {
		mGeneralEdit2Text = temp;
		theta = temp_float;
	}
}

// Open mat data button clicked
void RenderWindowUISingleInheritance::on_importButton_clicked()
{
	import = new ImageImportDialog();
	connect(import, SIGNAL(matFileSelected(QString, QString)), this, SLOT(fileSelected(QString, QString)));
    import->exec();
}

void RenderWindowUISingleInheritance::on_startButton_clicked()
{
	if (use_saved_instance)
	{
		sig = 's';
		this->dumpLog("Simulation start:", 0, 1);
	} else {
		if (mMatFilename != "" && mMatVarName != "") {			
			sig = 's';
		} else {
			/* warn that the .mat file is not selected */
			this->dumpLog("no .mat file selected or variable name incorrect.", 0, 1);
		}
	}
}

void RenderWindowUISingleInheritance::dumpLog(QString str, int mode, bool newline)
{
	if (newline) {
		this->ui->logEdit->moveCursor(QTextCursor::End);
		this->ui->logEdit->insertPlainText(QString("\n"));
	}
	this->ui->logEdit->moveCursor(QTextCursor::End);
	QString head;
	QString indicator;
	QString tail;
	QString temp;
	switch(mode) {
	case 0:
		head = "<font color=\"blue\">";
		indicator = ">>> ";
		tail = "</font>";
		temp = head + indicator + tail + str;
		break;
	case 1:
		head = "<font color=\"red\">";
		indicator = "";
		tail = "</font>";
		temp = head + indicator + str + tail;
		break;
	case 2:
		head = "<font color=\"blue\">";
		indicator = ">>> ";
		tail = "</font>";
		temp = head + indicator + tail + QString("<font color=\"green\">") + str + QString("</font>");
		break;
	}

	this->ui->logEdit->insertHtml(temp);
}

void RenderWindowUISingleInheritance::dumpLogStr(std::string input_str, int mode, bool newline)
{
	QString str = QString(input_str.c_str());
	if (newline) {
		this->ui->logEdit->moveCursor(QTextCursor::End);
		this->ui->logEdit->insertPlainText(QString("\n"));
	}
	this->ui->logEdit->moveCursor(QTextCursor::End);
	QString head;
	QString indicator;
	QString tail;
	QString temp;
	switch(mode) {
	case 0:
		head = "<font color=\"blue\">";
		indicator = ">>> ";
		tail = "</font>";
		temp = head + indicator + tail + str;
		break;
	case 1:
		head = "<font color=\"red\">";
		indicator = "";
		tail = "</font>";
		temp = head + indicator + str + tail;
	case 2:
		head = "<font color=\"blue\">";
		indicator = ">>> ";
		tail = "</font>";
		temp = head + indicator + tail + QString("<font color=\"green\">") + str + QString("</font>");
		break;
	}

	this->ui->logEdit->insertHtml(temp);
}

QString RenderWindowUISingleInheritance::getMatFilename()
{
	return mMatFilename;
}

QString RenderWindowUISingleInheritance::getMatVarName()
{
	return mMatVarName;
}

void RenderWindowUISingleInheritance::setMatFilename(QString filename)
{
	mMatFilename = filename;
}

float RenderWindowUISingleInheritance::getPhi()
{
	return phi;
}

float RenderWindowUISingleInheritance::getTheta()
{
	return theta;
}