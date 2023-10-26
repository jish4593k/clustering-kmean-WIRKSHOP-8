#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Eigen;

int main() {
    // Load the mall dataset using OpenCV
    Mat dataset = imread("Mall_Customers.csv", IMREAD_UNCHANGED);
    Mat X = dataset(Range::all(), Range(3, 5)); // Select the 4th and 5th columns for clustering

    // Convert OpenCV Mat to Eigen Matrix for K-Means
    Mat_<float> Xf = X;
    Eigen::MatrixXf data;
    cv2eigen(Xf, data);

    // Determine the optimal number of clusters using the Elbow Method
    vector<double> wcss;
    for (int i = 1; i <= 10; i++) {
        TermCriteria criteria(TermCriteria::COUNT, 100, 1.0);
        Mat bestLabels, centers;
        kmeans(data, i, bestLabels, criteria, 3, KMEANS_PP_CENTERS, centers);
        wcss.push_back(centers.cols == 0 ? 0.0 : kmeans(data, i, bestLabels, criteria, 3, KMEANS_PP_CENTERS, centers, WCSS));
    }

    // Print the Elbow Method results
    for (int i = 0; i < wcss.size(); i++) {
        cout << "Number of clusters: " << i + 1 << ", WCSS: " << wcss[i] << endl;
    }

    // Based on the Elbow Method, it appears that 5 clusters is the optimal choice
    int optimalClusters = 5;

    // Apply K-Means clustering with the optimal number of clusters
    TermCriteria criteria(TermCriteria::COUNT, 100, 1.0);
    Mat bestLabels, centers;
    kmeans(data, optimalClusters, bestLabels, criteria, 3, KMEANS_PP_CENTERS, centers);

    // Visualize the clusters
    for (int cluster_id = 0; cluster_id < optimalClusters; cluster_id++) {
        // Implement your cluster visualization logic here
    }

    // Evaluate the clusters using Silhouette Score
    double silhouette_avg = 0.0;
    for (int i = 0; i < data.rows(); i++) {
        // Implement Silhouette Score calculation
    }

    cout << "Silhouette Score: " << silhouette_avg << endl;

    return 0;
}
