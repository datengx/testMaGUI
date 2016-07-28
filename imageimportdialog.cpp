#include "imageimportdialog.h"
#include <QFileDialog.h>
#include <QMessageBox.h>
#include <iostream>
ImageImportDialog::ImageImportDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ImportDialog)
{
    ui->setupUi(this);

    // Set text editor
    QPalette *palette = new QPalette();
    palette->setColor(QPalette::Base,Qt::gray);
    palette->setColor(QPalette::Text,Qt::black);
    ui->imageEdit->setPalette(*palette); // Set line editors to be read only
    ui->imageEdit->setReadOnly(true);
    ui->matEdit->setPalette(*palette);
    ui->matEdit->setReadOnly(true);

    // Initialize filename
    matFilename = "";
    imageFilename = "";
}

ImageImportDialog::~ImageImportDialog()
{
    delete ui;
}

// Import image files
void ImageImportDialog::on_openImageButton_clicked()
{
    QString temp = QFileDialog::getOpenFileName(
                this,
                tr("Choose input image file"),
                "D://",
                "Image Files (*.jpg *.png *.pgm);; All Files (*.*)"// What kind of files to see by default

                );
    if (temp.contains(QRegExp(".+((\\.png$)|(\\.jpg$)|(\\.jpeg$)|(\\.pgm$)|(\\.mha$))"))) {
        std::cout << "valid image input" << std::endl;
        imageFilename = temp;
    }
    ui->imageEdit->setText(imageFilename);
}


// Import .mat files
void ImageImportDialog::on_openMatButton_clicked()

{
    QString temp = QFileDialog::getOpenFileName(
                this,
                tr("Choose input .mat file"),
                "D://RLSimulation//itk_projects//testMatGUi//build//input",
                ".MAT Files (*.mat)"// What kind of files to see by default
                );
    if (temp.contains(QRegExp(".+((\\.mat$))"))) {
        std::cout << "valid .mat input" << std::endl;
        matFilename = temp;
    }
    ui->matEdit->setText(matFilename);

}

void ImageImportDialog::on_saveButton_clicked()
{
    if (imageFilename != "" || matFilename != "") {
        emit matFileSelected(imageFilename, matFilename);
        std::cout << "After signal emitted" << std::endl;
    }
    close();
}
