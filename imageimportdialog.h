#ifndef IMAGEIMPORTDIALOG_H
#define IMAGEIMPORTDIALOG_H

#include <QDialog>
#include <ui_imageImportDialog.h>

namespace Ui {
    class ImportDialog;
}

class ImageImportDialog : public QDialog
{
    Q_OBJECT
public:
    explicit ImageImportDialog(QWidget *parent = 0);
    ~ImageImportDialog();

signals:
    void matFileSelected(QString s_image, QString s_mat);
public slots:

private slots:
    void on_openImageButton_clicked();

    void on_openMatButton_clicked();

    void on_saveButton_clicked();

private:
    Ui::ImportDialog *ui;
    QString imageFilename;
    QString matFilename;
};

#endif // IMAGEIMPORTDIALOG_H
