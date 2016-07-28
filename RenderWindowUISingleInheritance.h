#ifndef RenderWindowUISingleInheritance_H
#define RenderWindowUISingleInheritance_H
 
#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include "imageimportdialog.h"
#include <string>

#include <QMainWindow>
 
// Forward Qt class declarations
class Ui_RenderWindowUISingleInheritance;
 
class RenderWindowUISingleInheritance : public QMainWindow
{
  Q_OBJECT
public:
 
  // Constructor/Destructor
  RenderWindowUISingleInheritance();
  RenderWindowUISingleInheritance(vtkSmartPointer<vtkRenderer> renderer);
  ~RenderWindowUISingleInheritance() {};
  void updateQVTKWidget();
  void closeEvent(QCloseEvent* event);
  void dumpLog(QString str, int mode, bool newline);
  void dumpLogStr(std::string input_str, int mode, bool newline);
  QString getMatFilename();
  QString getMatVarName();
  float getPhi();
  float getTheta();
  void setMatFilename(QString);
public slots:
 
  virtual void slotExit();
  void fileSelected(QString s_image, QString s_mat);
  void onMatVarNameChanged();
  void onMatFileNameChanged();
  void onGeneralEdit1Changed();
  void onGeneralEdit2Changed();
  void importCheckBoxToggled(bool toggle);

private slots:
  void on_importButton_clicked();
  void on_startButton_clicked();
private:



  // Designer form
  Ui_RenderWindowUISingleInheritance *ui;
  ImageImportDialog* import;
  QString mMatFilename;
  QString mMatVarName;
  QString mGeneralEdit1Text;
  QString mGeneralEdit2Text;
  float phi;
  float theta;
};
 
#endif
