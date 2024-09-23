import sys
import pandas as pd
import pickle
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout,
    QWidget, QFormLayout, QGroupBox, QMessageBox, QHBoxLayout
)
from PyQt5.QtCore import Qt


class MaterialPredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.load_models()

    def initUI(self):
        self.setWindowTitle('材料性能预测')
        self.setGeometry(100, 100, 600, 400)

        # Create main layout
        main_layout = QVBoxLayout()

        # Create file operation group
        file_group = QGroupBox('文件操作')
        file_layout = QVBoxLayout()

        # Create and style buttons
        self.upload_btn = QPushButton('上传特征文件')
        self.upload_btn.setObjectName('upload_btn')
        self.upload_btn.setFixedSize(150, 60)  # Set button size

        self.predict_btn = QPushButton('进行预测')
        self.predict_btn.setObjectName('predict_btn')
        self.predict_btn.setFixedSize(150, 60)  # Set button size

        self.save_btn = QPushButton('保存预测结果')
        self.save_btn.setObjectName('save_btn')
        self.save_btn.setFixedSize(150, 60)  # Set button size

        # Create a function to center buttons
        def create_centered_layout(button):
            layout = QHBoxLayout()
            layout.addStretch(1)  # Add stretchable space before the button
            layout.addWidget(button)
            layout.addStretch(1)  # Add stretchable space after the button
            return layout

        # Add buttons to the layout with centered alignment
        file_layout.addLayout(create_centered_layout(self.upload_btn))
        file_layout.addLayout(create_centered_layout(self.predict_btn))
        file_layout.addLayout(create_centered_layout(self.save_btn))

        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)

        # Create result display group
        result_group = QGroupBox('预测结果')
        result_layout = QVBoxLayout()
        self.result_label = QLabel('预测结果将在此显示')
        result_layout.addWidget(self.result_label)
        result_group.setLayout(result_layout)
        main_layout.addWidget(result_group)

        # Set the main layout
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Apply styles
        self.setStyleSheet("""
            /* Background color */
            QWidget {
                background-color: #FFF9E1; /* Light yellow */
            }
            QPushButton {
                padding: 10px;
                margin: 5px;
                border-radius: 10px; /* Rounded corners */
                color: black;
            }
            QPushButton#upload_btn {
                background-color: #B6F1E9; /* Purple */
                font-size: 16px; /* Custom font size */
                font-weight: bold; /* Custom font weight */
            }
            QPushButton#predict_btn {
                background-color: #B6F1E9; /* Orange */
                font-size: 16px; /* Custom font size */
                font-weight: bold; /* Custom font weight */
            }
            QPushButton#save_btn {
                background-color: #B6F1E9; /* Blue */
                font-size: 16px; /* Custom font size */
                font-weight: bold; /* Custom font weight */
            }
            QLabel {
                font-size: 14px;
                font-weight: bolder;
                color: black;
            }
            QGroupBox {
                font-size: 18px;
                font-weight: bold;
                color: black;
            }
            QVBoxLayout {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #cfe8ff, stop:1 #e6f7ff);
            }
        """)

    def load_models(self):
        # Load models with pickle
        with open('model_pickle/GBR_ts_descriptor_selected.pkl', 'rb') as file:
            self.model_ts = pickle.load(file)

        with open('model_pickle/GBR_el_descriptor_selected.pkl', 'rb') as file:
            self.model_el = pickle.load(file)

        # 定义模型所需的特征名
        self.ts_features = ['ave:covalent_radius_pyykko_double', 'ave:hhi_p', 'ave:num_s_unfilled',
                            'sum:fusion_enthalpy', 'var:bulk_modulus', 'var:num_p_unfilled',
                            'var:num_s_unfilled', 'var:num_s_valence', 'var:vdw_radius_uff', 'TT',
                            'TT_time']  # 替换为实际特征名
        self.el_features = ['ave:covalent_radius_pyykko', 'ave:covalent_radius_pyykko_double',
                            'ave:en_allen', 'ave:vdw_radius_mm3', 'sum:dipole_polarizability',
                            'sum:fusion_enthalpy', 'sum:hhi_p', 'sum:hhi_r',
                            'sum:heat_of_formation', 'sum:num_unfilled', 'sum:num_s_unfilled',
                            'sum:specific_heat', 'var:bulk_modulus', 'var:hhi_p', 'var:num_valance',
                            'var:vdw_radius_uff', 'QT', 'TT', 'TT_time']

    def upload_file(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "选择特征文件", "", "CSV Files (*.csv);;All Files (*)",
                                                  options=options)
        if fileName:
            self.data = pd.read_csv(fileName)
            self.result_label.setText(f'已上传文件: {fileName}')

    def predict(self):
        if hasattr(self, 'data'):
            try:
                # Ensure that the required columns are present
                ts_data = self.data[self.ts_features]
                el_data = self.data[self.el_features]

                # Perform predictions
                ts_predictions = self.model_ts.predict(ts_data)
                el_predictions = self.model_el.predict(el_data)

                # Create a DataFrame to store results
                results_df = self.data.copy()
                results_df['TS_Prediction'] = ts_predictions
                results_df['EL_Prediction'] = el_predictions

                # Display a summary of the results
                summary = f'预测完成！共{len(results_df)}条数据'
                self.result_label.setText(summary)

                # Store results for saving
                self.results_df = results_df
            except KeyError as e:
                QMessageBox.warning(self, "错误", f"特征缺失: {e}")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"预测时发生错误: {e}")
        else:
            QMessageBox.warning(self, "警告", '请先上传特征文件')

    def save_results(self):
        if hasattr(self, 'results_df'):
            options = QFileDialog.Options()
            fileName, _ = QFileDialog.getSaveFileName(self, "保存预测结果", "", "CSV Files (*.csv);;All Files (*)",
                                                      options=options)
            if fileName:
                try:
                    self.results_df.to_csv(fileName, index=False)
                    QMessageBox.information(self, "成功", "预测结果已保存")
                except Exception as e:
                    QMessageBox.warning(self, "错误", f"保存文件时发生错误: {e}")
        else:
            QMessageBox.warning(self, "警告", "没有预测结果可保存")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MaterialPredictionApp()
    ex.show()
    sys.exit(app.exec_())
