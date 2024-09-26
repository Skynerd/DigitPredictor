using System;
using System.Collections.Generic;
using System.Data;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace Common
{
    public static class clsCommon
    {


        public static void SetPictureBox(this PictureBox pictureBox, Image image, Point? point = null)
        {
            if (point != null) pictureBox.Location = (Point)point;
            pictureBox.Image = image;
            pictureBox.Size = image.Size;
        }

        public static int[] ToIntArray(this string input, char split = '\n')
        {
            // Split the string by newlines and remove any empty entries
            string[] stringNumbers = input.Split(new[] { split }, StringSplitOptions.RemoveEmptyEntries);

            // Convert each substring to an integer
            int[] intArray = stringNumbers.Select(int.Parse).ToArray();

            return intArray;
        }

        public static int GetMaxValueIndex(this double[] array)
        {
            if (array == null || array.Length == 0)
            {
                throw new ArgumentException("Array cannot be null or empty.");
            }

            int maxIndex = 0;  // Initialize maxIndex as the first index
            double maxValue = array[0];  // Initialize maxValue as the first element

            // Loop through the array to find the max value and its index
            for (int i = 1; i < array.Length; i++)
            {
                if (array[i] > maxValue)
                {
                    maxValue = array[i];
                    maxIndex = i;
                }
            }

            return maxIndex;
        }







        /// <summary>
        /// Reads all text from a specified file.
        /// </summary>
        /// <param name="filePath">The path to the text file.</param>
        /// <returns>A string containing all the text in the file.</returns>
        public static string Read(string filePath)
        {
            if (File.Exists(filePath))
            {
                try
                {
                    return File.ReadAllText(filePath);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error reading file: {ex.Message}");
                    return string.Empty;
                }
            }
            else
            {
                Console.WriteLine("Error: File does not exist.");
                return string.Empty;
            }
        }

        /// <summary>
        /// Writes text to a specified file. If the file exists, it will be overwritten.
        /// </summary>
        /// <param name="filePath">The path to the text file.</param>
        /// <param name="content">The content to write to the file.</param>
        public static void Write(string filePath, string content)
        {
            try
            {
                File.WriteAllText(filePath, content);
                Console.WriteLine("File written successfully.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error writing to file: {ex.Message}");
            }
        }

        public static string Compress(this string data)
        {
            // Convert the JSON string to a byte array
            byte[] dataBytes = Encoding.UTF8.GetBytes(data);

            // Create a memory stream to hold the compressed data
            using (MemoryStream compressedStream = new MemoryStream())
            {
                // Create a GZipStream for compression
                using (GZipStream gzipStream = new GZipStream(compressedStream, CompressionMode.Compress))
                {
                    // Write the JSON bytes to the GZipStream
                    gzipStream.Write(dataBytes, 0, dataBytes.Length);
                }

                // Convert the compressed data to a Base64 string to make it easy to store and transmit
                return Convert.ToBase64String(compressedStream.ToArray());
            }
        }

        public static string Decompress(this string compressedData)
        {
            // Convert the Base64 encoded string back to a byte array
            byte[] compressedBytes = Convert.FromBase64String(compressedData);

            // Create a memory stream to hold the decompressed data
            using (MemoryStream compressedStream = new MemoryStream(compressedBytes))
            {
                // Create a GZipStream for decompression
                using (GZipStream gzipStream = new GZipStream(compressedStream, CompressionMode.Decompress))
                {
                    // Create a memory stream to hold the decompressed bytes
                    using (MemoryStream decompressedStream = new MemoryStream())
                    {
                        // Copy the decompressed data from the GZipStream to the memory stream
                        gzipStream.CopyTo(decompressedStream);

                        // Convert the decompressed bytes back to a string
                        return Encoding.UTF8.GetString(decompressedStream.ToArray());
                    }
                }
            }
        }

        public static string ToJson(this Dictionary<string, object> dictionary)
        {
            // Serialize the dictionary to a JSON string
            return JsonSerializer.Serialize(dictionary);
        }

        public static string ToJson(this NeuralNetwork NN)
        {
            // Serialize the dictionary to a JSON string
            return JsonSerializer.Serialize(NN);
        }
        public static Dictionary<string, object> ToDictionary(this string jsonString)
        {
            // Deserialize the JSON string to a Dictionary<string, object>
            return JsonSerializer.Deserialize<Dictionary<string, object>>(jsonString);
        }
        public static NeuralNetwork ToNeuralNetwork(this string jsonString)
        {
            // Deserialize the JSON string to a Dictionary<string, object>
            return JsonSerializer.Deserialize<NeuralNetwork>(jsonString);
        }

        public static int[] Parse(string input)
        {
            if (string.IsNullOrWhiteSpace(input))
            {
                // Return an empty array if the input is null, empty, or whitespace
                return new int[0];
            }

            // Split the input string by commas and optionally by spaces
            string[] parts = input.Trim('[', ']').Split(new char[] { ',', ' ' }, StringSplitOptions.RemoveEmptyEntries);

            List<int> numbers = new List<int>();

            foreach (string part in parts)
            {
                if (int.TryParse(part.Trim(), out int number))
                {
                    // Add the parsed integer to the list
                    numbers.Add(number);
                }
                else
                {
                    // If parsing fails, throw an exception or handle as needed
                    throw new FormatException($"Unable to parse '{part}' as an integer.");
                }
            }

            // Convert the list to an array and return it
            return numbers.ToArray();
        }



 
    }


    public static class MsgBox
    {
        public static void Error(string message)
        {
            MessageBox.Show(
                "Error: " + message,
                "Error",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error
            );
        }
        public static void Warning(string message)
        {
            MessageBox.Show(
                "Warning: " + message,
                "Warning",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error
            );
        }
        public static void Information(string message)
        {
            MessageBox.Show(
                "Info: " + message,
                "Information",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error
            );
        }
    }




    public class NeuralNetwork
    {
        public int[] Structure { get; set; }
        public Matrix[] Weights { get; set; }
        public Vector[] Biases { get; set; }
        //public double LearningRate { get; set; }


        public NeuralNetwork(int[] dataStructure, double learningRate = 0.01)
        {
            // Structure
            Structure = dataStructure;

            // LearningRate
            //LearningRate = learningRate;

            // Weights
            Weights = new Matrix[Structure.Length - 1];
            for (int i = 0; i < Structure.Length - 1; i++)
            {
                Weights[i] = new Matrix(Structure[i + 1], Structure[i]).Randomize();
            }

            // Biases
            Biases = new Vector[Structure.Length - 1];
            for (int i = 0; i < Structure.Length - 1; i++)
            {
                Biases[i] = new Vector(Structure[i + 1]).Randomize();
            }

        }
        // Constructor with JsonConstructor attribute
        [JsonConstructor]
        public NeuralNetwork(int[] structure, Matrix[] weights, Vector[] biases)
        {
            Structure = structure;
            Weights = weights;
            Biases = biases;
        }

        public static NeuralNetwork Load(string filename, bool compressed = true)
        {
            if (compressed) return File.ReadAllText(filename).Decompress().ToNeuralNetwork();
            return File.ReadAllText(filename).ToNeuralNetwork();
        }




        // Parse method to convert a string to a Vector object
        public static NeuralNetwork Parse(string input)
        {
            if (TryParse(input, out NeuralNetwork neuralNetwork))
            {
                return neuralNetwork;
            }
            else
            {
                throw new FormatException("Input string is not in the correct format for a Vector.");
            }
        }

        // TryParse method to safely convert a string to a Vector object
        public static bool TryParse(string input, out NeuralNetwork result)
        {
            result = null;
            if (string.IsNullOrWhiteSpace(input))
            {
                return false;
            }

            input = input.Trim('[', ']');
            string[] parts = input.Split(',');

            int[] data = new int[parts.Length];
            for (int i = 0; i < parts.Length; i++)
            {
                if (!int.TryParse(parts[i].Trim(), out data[i]))
                {
                    return false;
                }
            }

            result = new NeuralNetwork(data);
            return true;
        }



        // Forward pass function
        public Vector Forward(Vector input)
        {
            Vector output = input;

            // Apply ReLU activation function to each hidden layer
            for (int i = 0; i < Weights.Length - 1; i++)
            {
                output = ReLU((Weights[i] * output) + Biases[i]);
            }

            // Apply sigmoid activation function to the output layer
            output = Sigmoid((Weights[Weights.Length - 1] * output) + Biases[Biases.Length - 1]);

            return output;
        }

        // ReLU activation function
        private Vector ReLU(Vector vector)
        {
            Vector result = new Vector(vector.Length);
            for (int i = 0; i < vector.Length; i++)
            {
                result.Data[i] = Math.Max(0, vector.Data[i]);
            }
            return result;
        }

        // Sigmoid activation function
        private Vector Sigmoid(Vector vector)
        {
            Vector result = new Vector(vector.Length);
            for (int i = 0; i < vector.Length; i++)
            {
                result[i] = 1.0 / (1.0 + Math.Exp(-vector[i]));
            }
            return result;
        }

        // Derivative of the Sigmoid function
        private Vector SigmoidDerivative(Vector vector)
        {
            Vector result = new Vector(vector.Length);
            for (int i = 0; i < vector.Length; i++)
            {
                double sigmoidValue = 1.0 / (1.0 + Math.Exp(-vector[i]));
                result[i] = sigmoidValue * (1 - sigmoidValue);
            }
            return result;
        }

        // Derivative of the ReLU function
        private Vector ReLUDerivative(Vector vector)
        {
            Vector result = new Vector(vector.Length);
            for (int i = 0; i < vector.Length; i++)
            {
                result.Data[i] = vector.Data[i] > 0 ? 1 : 0;
            }
            return result;
        }

        // Backpropagation function
        public Vector Backpropagate(Vector input, Vector expectedOutput, double LearningRate = 0.01)
        {
            // Step 1: Forward pass
            Vector[] activations = new Vector[Structure.Length];
            Vector[] zs = new Vector[Structure.Length - 1]; // z is the weighted input
            activations[0] = input;

            for (int i = 0; i < Weights.Length; i++)
            {
                zs[i] = (Weights[i] * activations[i]) + Biases[i];
                activations[i + 1] = (i == Weights.Length - 1) ? Sigmoid(zs[i]) : ReLU(zs[i]);
            }

            // Step 2: Compute output error
            Vector output = activations[^1]; // Last layer output
            Vector delta = (output - expectedOutput) * SigmoidDerivative(output);

            // Initialize gradient storage
            Vector[] nablaB = new Vector[Biases.Length];
            Matrix[] nablaW = new Matrix[Weights.Length];

            for (int i = 0; i < nablaB.Length; i++)
            {
                nablaB[i] = new Vector(Biases[i].Length);
                nablaW[i] = new Matrix(Weights[i].Rows, Weights[i].Columns);
            }

            // Step 3: Backpropagate error
            nablaB[^1] = delta;
            nablaW[^1] = delta.Cross(activations[^2]);

            for (int l = 2; l < Structure.Length; l++)
            {
                Vector z = zs[^l];
                Vector sp = ReLUDerivative(z);
                delta = (Weights[Weights.Length - l + 1].Transpose() * delta) * sp;
                nablaB[^l] = delta;
                nablaW[^l] = delta.Cross(activations[activations.Length - l - 1]);
            }

            // Step 4: Update weights and biases
            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = Weights[i] - (nablaW[i] * LearningRate);
                Biases[i] = Biases[i] - (nablaB[i] * LearningRate);
            }

            return (output - expectedOutput) / Math.Sqrt(expectedOutput.Length);

        }
    }


    public class Matrix
    {
        public double[][] Data { get; set; }
        public int Rows { get; }
        public int Columns { get; }

        // Constructor to initialize a matrix of given dimensions
        public Matrix(int rows, int columns)
        {
            Rows = rows;
            Columns = columns;
            Data = new double[rows][];
            for (int i = 0; i < rows; i++)
            {
                Data[i] = new double[columns];
            }
        }

        // Constructor to initialize a matrix of given dimensions
        [JsonConstructor]
        public Matrix(double[][] data)
        {
            Rows = data.Length;
            Columns = data[0].Length;
            Data = data;
        }

        // Indexer to access matrix elements
        public double this[int row, int column]
        {
            get => Data[row][column];
            set => Data[row][column] = value;
        }

        // Overload the '*' operator to perform matrix multiplication using parallel processing
        public static Matrix operator *(Matrix matrixA, Matrix matrixB)
        {
            if (matrixA.Columns != matrixB.Rows)
                throw new InvalidOperationException("Number of columns in Matrix A must be equal to the number of rows in Matrix B.");

            Matrix result = new Matrix(matrixA.Rows, matrixB.Columns);

            Parallel.For(0, matrixA.Rows, i =>
            {
                for (int j = 0; j < matrixB.Columns; j++)
                {
                    for (int k = 0; k < matrixA.Columns; k++)
                    {
                        result[i, j] += matrixA[i, k] * matrixB[k, j];
                    }
                }
            });

            return result;
        }

        // Overload the '+' operator to perform matrix addition
        public static Matrix operator +(Matrix matrixA, Matrix matrixB)
        {
            if (matrixA.Rows != matrixB.Rows || matrixA.Columns != matrixB.Columns)
                throw new InvalidOperationException("Matrices must have the same dimensions for addition.");

            Matrix result = new Matrix(matrixA.Rows, matrixA.Columns);

            Parallel.For(0, matrixA.Rows, i =>
            {
                for (int j = 0; j < matrixA.Columns; j++)
                {
                    result[i, j] = matrixA[i, j] + matrixB[i, j];
                }
            });

            return result;
        }

        // Overload the '-' operator to perform matrix addition
        public static Matrix operator -(Matrix matrixA, Matrix matrixB)
        {
            if (matrixA.Rows != matrixB.Rows || matrixA.Columns != matrixB.Columns)
                throw new InvalidOperationException("Matrices must have the same dimensions for addition.");

            Matrix result = new Matrix(matrixA.Rows, matrixA.Columns);

            Parallel.For(0, matrixA.Rows, i =>
            {
                for (int j = 0; j < matrixA.Columns; j++)
                {
                    result[i, j] = matrixA[i, j] - matrixB[i, j];
                }
            });

            return result;
        }

        // Overload the '*' operator to perform matrix-vector multiplication
        public static Vector operator *(Matrix matrix, Vector vector)
        {
            if (matrix.Columns != vector.Length)
                throw new InvalidOperationException("Number of columns in the matrix must equal the length of the vector.");

            Vector result = new Vector(matrix.Rows);

            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    result[i] += matrix[i, j] * vector[j];
                }
            }

            return result;
        }

        // Multiply a matrix by a scalar
        public static Matrix operator *(Matrix matrix, double scalar)
        {
            Matrix result = new Matrix(matrix.Rows, matrix.Columns);

            // Parallel processing to multiply each element by the scalar
            Parallel.For(0, matrix.Rows, i =>
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    result[i, j] = matrix[i, j] * scalar;
                }
            });

            return result;
        }

        // Transpose function
        public Matrix Transpose()
        {
            Matrix transposedMatrix = new Matrix(this.Columns, this.Rows);

            // Parallel processing to transpose the matrix
            Parallel.For(0, this.Rows, i =>
            {
                for (int j = 0; j < this.Columns; j++)
                {
                    transposedMatrix[j, i] = this[i, j];
                }
            });

            return transposedMatrix;
        }


        // Convert the matrix to a string representation
        public override string ToString()
        {
            string[] rows = new string[Rows];
            for (int i = 0; i < Rows; i++)
            {
                string[] columns = new string[Columns];
                for (int j = 0; j < Columns; j++)
                {
                    columns[j] = Data[i][j].ToString("F2");
                }
                rows[i] = "[" + string.Join(", ", columns) + "]";
            }
            return "[" + string.Join("; ", rows) + "]";
        }

        // Parse method to convert a string to a Matrix object
        public static Matrix Parse(string input)
        {
            if (TryParse(input, out Matrix matrix))
            {
                return matrix;
            }
            else
            {
                throw new FormatException("Input string is not in the correct format for a Matrix.");
            }
        }

        // TryParse method to safely convert a string to a Matrix object
        public static bool TryParse(string input, out Matrix result)
        {
            result = null;
            if (string.IsNullOrWhiteSpace(input))
            {
                return false;
            }

            input = input.Trim('[', ']');
            string[] rows = input.Split(';');

            if (rows.Length == 0)
            {
                return false;
            }

            int numColumns = rows[0].Split(',').Length;
            result = new Matrix(rows.Length, numColumns);

            for (int i = 0; i < rows.Length; i++)
            {
                string[] columns = rows[i].Split(',');
                if (columns.Length != numColumns)
                {
                    return false;  // inconsistent column length
                }

                for (int j = 0; j < columns.Length; j++)
                {
                    if (!double.TryParse(columns[j].Trim(), out double value))
                    {
                        return false;
                    }
                    result[i, j] = value;
                }
            }

            return true;
        }



        public Matrix Randomize()
        {
            // Thread-safe random number generator for parallel operations
            var random = new ThreadLocal<Random>(() => new Random());

            // Use Parallel.For to fill the matrix with random values
            Parallel.For(0, this.Rows, i =>
            {
                for (int j = 0; j < this.Columns; j++)
                {
                    // Populate matrix with random double values
                    this[i, j] = random.Value.NextDouble(); // Generates a random double between 0.0 and 1.0
                }
            });

            return this;
        }


    }



    public class Vector
    {
        public double[] Data { get; }
        public int Length { get; }

        // Constructor to initialize a vector of given size
        public Vector(int length)
        {
            Length = length;
            Data = new double[length];
        }

        // Constructor to initialize a vector with an array of values
        [JsonConstructor]
        public Vector(double[] data)
        {
            Data = data;
            Length = data.Length;
        }

        //// Constructor to initialize a vector with an array of values
        //public Vector(int[] data)
        //{
        //    Data = (double[])data;
        //}

        // Indexer to access vector elements
        public double this[int index]
        {
            get => Data[index];
            set => Data[index] = value;
        }

        // Overload the '+' operator to perform vector addition using Parallel.For
        public static Vector operator +(Vector vectorA, Vector vectorB)
        {
            if (vectorA.Length != vectorB.Length)
                throw new InvalidOperationException("Vectors must have the same length for addition.");

            Vector result = new Vector(vectorA.Length);
            Parallel.For(0, vectorA.Length, i =>
            {
                result[i] = vectorA[i] + vectorB[i];
            });

            return result;
        }

        // Overload the '-' operator to perform vector addition using Parallel.For
        public static Vector operator -(Vector vectorA, Vector vectorB)
        {
            if (vectorA.Length != vectorB.Length)
                throw new ArgumentException("Vectors must be of the same length.");

            Vector result = new Vector(vectorA.Length);
            Parallel.For(0, vectorA.Length, i =>
            {
                result[i] = vectorA[i] - vectorB[i];
            });

            return result;
        }

        // Overload the '*' operator to perform dot product
        public static Vector operator *(Vector vector, double scalar)
        {
            Vector result = new Vector(vector.Length);
            for (int i = 0; i < vector.Length; i++)
            {
                result[i] = vector[i] * scalar;
            }

            return result;
        }

        // Overload the '*' operator to perform dot product
        public static Vector operator /(Vector vector, double scalar)
        {
            Vector result = new Vector(vector.Length);
            for (int i = 0; i < vector.Length; i++)
            {
                result[i] = vector[i] / scalar;
            }

            return result;
        }

        // Overload the '*' operator to perform dot product
        public static Vector operator *(Vector vectorA, Vector vectorB)
        {
            if (vectorA.Length != vectorB.Length)
                throw new InvalidOperationException("Vectors must have the same length for dot product.");

            Vector result = new Vector(vectorA.Length);
            for (int i = 0; i < vectorA.Length; i++)
            {
                result[i] = vectorA[i] * vectorB[i];
            }

            return result;
        }

        // Utility method to print the vector (optional)
        public override string ToString()
        {
            return "[" + string.Join(", ", Data) + "]";
        }

        public Matrix Cross(Vector vector)
        {
            Matrix result = new Matrix(this.Length, vector.Length);
            Parallel.For(0, this.Length, i =>
            {
                for (int j = 0; j < vector.Length; j++)
                {
                    result[i, j] = this[i] * vector[j];
                }
            });

            return result;
        }


        // Parse method to convert a string to a Vector object
        public static Vector Parse(string input)
        {
            if (TryParse(input, out Vector vector))
            {
                return vector;
            }
            else
            {
                throw new FormatException("Input string is not in the correct format for a Vector.");
            }
        }

        // TryParse method to safely convert a string to a Vector object
        public static bool TryParse(string input, out Vector result)
        {
            result = null;
            if (string.IsNullOrWhiteSpace(input))
            {
                return false;
            }

            input = input.Trim('[', ']');
            string[] parts = input.Split(',');

            double[] data = new double[parts.Length];
            for (int i = 0; i < parts.Length; i++)
            {
                if (!double.TryParse(parts[i].Trim(), out data[i]))
                {
                    return false;
                }
            }

            result = new Vector(data);
            return true;
        }



        public Vector Randomize()
        {
            // Thread-safe random number generator for parallel operations
            var random = new ThreadLocal<Random>(() => new Random());

            // Use Parallel.For to fill the vector with random values
            Parallel.For(0, this.Length, i =>
            {
                this[i] = random.Value.NextDouble(); // Generates a random double between 0.0 and 1.0 
            });

            return this;
        }





        public static Vector FromImage(Bitmap bitmap)
        {
            Vector vector = new Vector(bitmap.Width * bitmap.Height);

            // Loop through each pixel in the image
            for (int i = 0; i < bitmap.Height; i++)
            {
                for (int j = 0; j < bitmap.Height; j++)
                {
                    // Get the pixel color
                    Color pixel = bitmap.GetPixel(j, i);

                    // Since it's grayscale, R, G, and B should be the same.
                    // We can just use the R value for simplicity.
                    vector[i * bitmap.Width + j] = pixel.R / 255.0; // Normalize to [0, 1]
                }
            }

            return vector;
        }

        public static Vector FromImage(string imagePath)
        {
            Bitmap bitmap = new Bitmap(imagePath);

            Vector vector = new Vector(bitmap.Width * bitmap.Height);

            // Loop through each pixel in the image
            for (int i = 0; i < bitmap.Height; i++)
            {
                for (int j = 0; j < bitmap.Height; j++)
                {
                    // Get the pixel color
                    Color pixel = bitmap.GetPixel(j, i);

                    // Since it's grayscale, R, G, and B should be the same.
                    // We can just use the R value for simplicity.
                    vector[i * bitmap.Width + j] = pixel.R / 255.0; // Normalize to [0, 1]
                }
            }

            return vector;
        }

        /// <summary>
        /// Calculates the magnitude (length) of the vector.
        /// </summary>
        /// <returns>The magnitude of the vector.</returns>
        public double Magnitude()
        {
            double sumOfSquares = 0.0;

            // Sum the squares of each component
            foreach (double component in Data)
            {
                sumOfSquares += component * component;
            }

            // Return the square root of the sum of squares
            return Math.Sqrt(sumOfSquares);
        }


    }




    #region Img

    public static class Img
    {
        public static Image ResizeImage(this Image imgToResize, int x, int y)
        {
            try
            {
                Bitmap b = new Bitmap(x, y);
                using (Graphics g = Graphics.FromImage((Image)b))
                {
                    g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
                    g.DrawImage(imgToResize, 0, 0, x, y);
                }
                return b;
            }
            catch
            {
                Console.WriteLine("Bitmap could not be resized");
                return imgToResize;
            }
        }

        public static Image ResizeImage(Image imgToResize, Size size)
        {
            try
            {
                Bitmap b = new Bitmap(size.Width, size.Height);
                using (Graphics g = Graphics.FromImage((Image)b))
                {
                    g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
                    g.DrawImage(imgToResize, 0, 0, size.Width, size.Height);
                }
                return b;
            }
            catch
            {
                Console.WriteLine("Bitmap could not be resized");
                return imgToResize;
            }
        }

        public static Bitmap DrawFilledRectangle(Size size, Brush brushes = null)
        {
            if (brushes == null) brushes = Brushes.White;
            Bitmap bmp = new Bitmap(size.Width, size.Height);
            using (Graphics graph = Graphics.FromImage(bmp))
            {
                Rectangle ImageSize = new Rectangle(0, 0, size.Width, size.Height);
                graph.FillRectangle(brushes, ImageSize);
            }
            return bmp;
        }

        public static Bitmap DrawFilledRectangle(int x, int y, Brush brushes = null)
        {
            if (brushes == null) brushes = Brushes.White;
            Bitmap bmp = new Bitmap(x, y);
            using (Graphics graph = Graphics.FromImage(bmp))
            {
                Rectangle ImageSize = new Rectangle(0, 0, x, y);
                graph.FillRectangle(brushes, ImageSize);
            }
            return bmp;
        }


    }

    #endregion


    #region HID
    public static class KeyBoard
    {
        // Function 
        public static bool Enter = false;
        public static bool Ctrl = false;
        public static bool Shift = false;
        public static bool Tab = false;
        public static bool Del = false;
        public static bool Alt = false;

        // Alphabet
        public static bool A = false;
        public static bool B = false;
        public static bool C = false;
        public static bool D = false;
        public static bool E = false;
        public static bool F = false;
        public static bool G = false;
        public static bool H = false;
        public static bool I = false;
        public static bool J = false;
        public static bool K = false;
        public static bool L = false;
        public static bool M = false;
        public static bool N = false;
        public static bool O = false;
        public static bool P = false;
        public static bool Q = false;
        public static bool R = false;
        public static bool S = false;
        public static bool T = false;
        public static bool U = false;
        public static bool V = false;
        public static bool W = false;
        public static bool X = false;
        public static bool Y = false;
        public static bool Z = false;

        // Number
        public static bool no0 = false;
        public static bool no1 = false;
        public static bool no2 = false;
        public static bool no3 = false;
        public static bool no4 = false;
        public static bool no5 = false;
        public static bool no6 = false;
        public static bool no7 = false;
        public static bool no8 = false;
        public static bool no9 = false;

        public static void SetKeyboard(Keys keys, bool IsPressed)
        {
            switch (keys)
            {
                case Keys.Enter:
                    KeyBoard.Enter = IsPressed;
                    break;
                case Keys.ControlKey:
                    KeyBoard.Ctrl = IsPressed;
                    break;
                case Keys.ShiftKey:
                    KeyBoard.Shift = IsPressed;
                    break;
                case Keys.Tab:
                    KeyBoard.Tab = IsPressed;
                    break;
                case Keys.Delete:
                    KeyBoard.Del = IsPressed;
                    break;
                case Keys.Alt:
                    KeyBoard.Alt = IsPressed;
                    break;
                case Keys.A:
                    KeyBoard.A = IsPressed;
                    break;
                case Keys.B:
                    KeyBoard.B = IsPressed;
                    break;
                case Keys.C:
                    KeyBoard.C = IsPressed;
                    break;
                case Keys.D:
                    KeyBoard.D = IsPressed;
                    break;
                case Keys.E:
                    KeyBoard.E = IsPressed;
                    break;
                case Keys.F:
                    KeyBoard.F = IsPressed;
                    break;
                case Keys.G:
                    KeyBoard.G = IsPressed;
                    break;
                case Keys.H:
                    KeyBoard.H = IsPressed;
                    break;
                case Keys.I:
                    KeyBoard.I = IsPressed;
                    break;
                case Keys.J:
                    KeyBoard.J = IsPressed;
                    break;
                case Keys.K:
                    KeyBoard.K = IsPressed;
                    break;
                case Keys.L:
                    KeyBoard.L = IsPressed;
                    break;
                case Keys.M:
                    KeyBoard.M = IsPressed;
                    break;
                case Keys.N:
                    KeyBoard.N = IsPressed;
                    break;
                case Keys.O:
                    KeyBoard.O = IsPressed;
                    break;
                case Keys.P:
                    KeyBoard.P = IsPressed;
                    break;
                case Keys.Q:
                    KeyBoard.Q = IsPressed;
                    break;
                case Keys.R:
                    KeyBoard.R = IsPressed;
                    break;
                case Keys.S:
                    KeyBoard.S = IsPressed;
                    break;
                case Keys.T:
                    KeyBoard.T = IsPressed;
                    break;
                case Keys.U:
                    KeyBoard.U = IsPressed;
                    break;
                case Keys.V:
                    KeyBoard.V = IsPressed;
                    break;
                case Keys.W:
                    KeyBoard.W = IsPressed;
                    break;
                case Keys.X:
                    KeyBoard.X = IsPressed;
                    break;
                case Keys.Y:
                    KeyBoard.Y = IsPressed;
                    break;
                case Keys.Z:
                    KeyBoard.Z = IsPressed;
                    break;
                case Keys.LWin:
                    break;
                case Keys.RWin:
                    break;
                case Keys.NumPad0:
                    break;
                case Keys.NumPad1:
                    break;
                case Keys.NumPad2:
                    break;
                case Keys.NumPad3:
                    break;
                case Keys.NumPad4:
                    break;
                case Keys.NumPad5:
                    break;
                case Keys.NumPad6:
                    break;
                case Keys.NumPad7:
                    break;
                case Keys.NumPad8:
                    break;
                case Keys.NumPad9:
                    break;
                case Keys.Multiply:
                    break;
                case Keys.Add:
                    break;
                case Keys.Separator:
                    break;
                case Keys.Subtract:
                    break;
                case Keys.Decimal:
                    break;
                case Keys.Divide:
                    break;
                case Keys.F1:
                    break;
                case Keys.F2:
                    break;
                case Keys.F3:
                    break;
                case Keys.F4:
                    break;
                case Keys.F5:
                    break;
                case Keys.F6:
                    break;
                case Keys.F7:
                    break;
                case Keys.F8:
                    break;
                case Keys.F9:
                    break;
                case Keys.F10:
                    break;
                case Keys.F11:
                    break;
                case Keys.F12:
                    break;
                default:
                    break;
            }
        }
    }

    public class MouseCursor
    {
        public MouseState mouseState;
        public Cursor[]? cursors;
        public Point curPosMouse;
        public Point lastPosMouse;
        public int size;
        public Color color1;
        public Color color2;

        public MouseCursor()
        {
            mouseState = MouseState.Default;
            lastPosMouse = new Point(0, 0);
            color1 = Color.Black;
            color2 = Color.White;
            size = 1;
        }
    }

    public enum MouseState
    {
        Default,
        Hand,
        Pencil,
        Fill,
        Text,
        Eraser,
        Dropper,
        Zoom,
    }
    #endregion

}
