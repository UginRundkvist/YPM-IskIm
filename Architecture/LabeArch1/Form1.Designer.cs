namespace SortingLab
{
    partial class Form1
    {
        /// <summary>
        /// Обязательная переменная конструктора.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Освободить все используемые ресурсы.
        /// </summary>
        /// <param name="disposing">истинно, если управляемый ресурс должен быть удален; иначе ложно.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Код, автоматически созданный конструктором форм Windows

        /// <summary>
        /// Требуемый метод для поддержки конструктора — не изменяйте 
        /// содержимое этого метода с помощью редактора кода.
        /// </summary>
        private void InitializeComponent()
        {
            this.btnGenerate = new System.Windows.Forms.Button();
            this.btnRun = new System.Windows.Forms.Button();
            this.btnSave = new System.Windows.Forms.Button();
            this.btnLoad = new System.Windows.Forms.Button();
            this.dgvResults = new System.Windows.Forms.DataGridView();
            this.progressBar = new System.Windows.Forms.ProgressBar();
            this.txtLog = new System.Windows.Forms.RichTextBox();
            this.lblStatus = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.dgvResults)).BeginInit();
            this.SuspendLayout();
            // 
            // btnGenerate
            // 
            this.btnGenerate.Location = new System.Drawing.Point(12, 12);
            this.btnGenerate.Name = "btnGenerate";
            this.btnGenerate.Size = new System.Drawing.Size(75, 23);
            this.btnGenerate.TabIndex = 0;
            this.btnGenerate.Text = "Сгенерировать";
            this.btnGenerate.UseVisualStyleBackColor = true;
            // 
            // btnRun
            // 
            this.btnRun.Location = new System.Drawing.Point(109, 12);
            this.btnRun.Name = "btnRun";
            this.btnRun.Size = new System.Drawing.Size(75, 23);
            this.btnRun.TabIndex = 1;
            this.btnRun.Text = "Запустить";
            this.btnRun.UseVisualStyleBackColor = true;
            // 
            // btnSave
            // 
            this.btnSave.Location = new System.Drawing.Point(12, 55);
            this.btnSave.Name = "btnSave";
            this.btnSave.Size = new System.Drawing.Size(75, 23);
            this.btnSave.TabIndex = 2;
            this.btnSave.Text = "Созранить";
            this.btnSave.UseVisualStyleBackColor = true;
            // 
            // btnLoad
            // 
            this.btnLoad.Location = new System.Drawing.Point(109, 55);
            this.btnLoad.Name = "btnLoad";
            this.btnLoad.Size = new System.Drawing.Size(75, 23);
            this.btnLoad.TabIndex = 3;
            this.btnLoad.Text = "Загрузить";
            this.btnLoad.UseVisualStyleBackColor = true;
            // 
            // dgvResults
            // 
            this.dgvResults.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.dgvResults.Location = new System.Drawing.Point(12, 141);
            this.dgvResults.Name = "dgvResults";
            this.dgvResults.Size = new System.Drawing.Size(467, 251);
            this.dgvResults.TabIndex = 4;
            // 
            // progressBar
            // 
            this.progressBar.Location = new System.Drawing.Point(283, 55);
            this.progressBar.Name = "progressBar";
            this.progressBar.Size = new System.Drawing.Size(100, 23);
            this.progressBar.TabIndex = 6;
            // 
            // txtLog
            // 
            this.txtLog.Location = new System.Drawing.Point(514, 141);
            this.txtLog.Name = "txtLog";
            this.txtLog.Size = new System.Drawing.Size(488, 251);
            this.txtLog.TabIndex = 7;
            this.txtLog.Text = "";
            // 
            // lblStatus
            // 
            this.lblStatus.AutoSize = true;
            this.lblStatus.Location = new System.Drawing.Point(283, 36);
            this.lblStatus.Name = "lblStatus";
            this.lblStatus.Size = new System.Drawing.Size(47, 13);
            this.lblStatus.TabIndex = 8;
            this.lblStatus.Text = "lblStatus";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1095, 450);
            this.Controls.Add(this.lblStatus);
            this.Controls.Add(this.txtLog);
            this.Controls.Add(this.progressBar);
            this.Controls.Add(this.dgvResults);
            this.Controls.Add(this.btnLoad);
            this.Controls.Add(this.btnSave);
            this.Controls.Add(this.btnRun);
            this.Controls.Add(this.btnGenerate);
            this.Name = "Form1";
            this.Text = "Form1";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.dgvResults)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button btnGenerate;
        private System.Windows.Forms.Button btnRun;
        private System.Windows.Forms.Button btnSave;
        private System.Windows.Forms.Button btnLoad;
        private System.Windows.Forms.DataGridView dgvResults;
        //private System.Windows.Forms.Label lblStatus;
        private System.Windows.Forms.ProgressBar progressBar;
        private System.Windows.Forms.RichTextBox txtLog;
        private System.Windows.Forms.Label lblStatus;
    }
}

