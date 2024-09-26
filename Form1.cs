using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq; 
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Common;
using Microsoft.VisualBasic.Devices;


namespace DigitPredictorAI
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        // Canvas
        Image imgCanvas;

        // mouse
        public MouseCursor mouse = new MouseCursor(); 

        // DataBase
        NeuralNetwork modelNN;

        // paths
        string curPath = Directory.GetCurrentDirectory();



        private void frmDigit_Load(object sender, EventArgs e)
        {
            ClearCanvas();
            modelNN = NeuralNetwork.Load(curPath + "\\Models\\model.nn");//clsDA.LoadNN(18); 
        }

        private void frmDigit_Resize(object sender, EventArgs e)
        {
            ClearCanvas();
        }

        private void pbDraw_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left) { DrawIt(e, Color.White); Predict(); }

            if (mouse.lastPosMouse != e.Location) { mouse.lastPosMouse = e.Location; }
        }



        void DrawIt(MouseEventArgs e, Color color)
        {
            Brush brush = new SolidBrush(color);
            Point pastloc = mouse.lastPosMouse;
            Point loc = e.Location;
            float size = mouse.size * (float)numericUpDown1.Value;
            Pen pen = new Pen(color, size * 1.414f);
            pen.Alignment = System.Drawing.Drawing2D.PenAlignment.Center;
            using (Graphics graph = Graphics.FromImage(imgCanvas))
            {
                graph.DrawLine(pen, loc.X, loc.Y, pastloc.X, pastloc.Y);
                graph.FillEllipse(brush, loc.X - size / 2, loc.Y - size / 2, size, size);
            }
            pen.Dispose();

            pbDraw.SetPictureBox(Img.ResizeImage(imgCanvas, imgCanvas.Width, imgCanvas.Height));

            brush.Dispose();
        }

        void Predict()
        {
            Vector input = Vector.FromImage((Bitmap)imgCanvas.ResizeImage(28, 28));
            Vector output = modelNN.Forward(input);
            int idxAns = output.Data.GetMaxValueIndex();
            label1.Text = "";
            for (int i = 0; i < output.Length; i++)
            {
                label1.Text += i.ToString() + " - " + output[i].ToString("N3") + "\n";
            }
            label1.Text += "Ans: " + idxAns.ToString();
            label1.Text = label1.Text.Trim();
            label1.Update();
        }



        void ClearCanvas()
        {
            imgCanvas = Img.DrawFilledRectangle(pbDraw.Width, pbDraw.Height, Brushes.Black);
            pbDraw.SetPictureBox(imgCanvas, new Point(0, 0));
        }

        private void btnClear_Click(object sender, EventArgs e)
        {
            ClearCanvas();
        }
    }
}
