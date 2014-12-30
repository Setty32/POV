/**
 * File: main.cpp
 * Project: Panorama / POV
 * Author: Vojtech Kaisler, xkaisl00@stud.fit.vutbr.cz
 * Description: First attempt to create program which compose photos to panorama.
 * Used technology: SURF, RANSAC, homography, openCV, image wrapping
 *
 */

#define VISUAL_DEBUG 0

#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/stitching/warpers.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <cxcore.h>
#include <opencv/cxcore.h>
#include <iostream>
#include <sstream>
#include <cassert>
#include <algorithm>

#include <math.h>

#include <vector>

#define PI 3.14159265359

using namespace std;
using namespace cv;

float maxDimensionsWithRatio(int &width, int &height, int maxSize) {
    float scale = 1.0f;
    if (width > maxSize || height > maxSize) {
        if (width > height) {
            scale = (float(width) / float(height));
            width = maxSize;
            height = maxSize * scale;
        } else {
            scale = float(height) / float(width);
            width = maxSize * scale;
            height = maxSize;
        }
    }
    return scale;
}

void imshowScaled(const string& winname, InputArray mat, int maxSize = 1200) {

    namedWindow(winname, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);

    imshow(winname, mat);

    Mat m = mat.getMat();
    int w, h;

    w = m.cols;
    h = m.rows;

    maxDimensionsWithRatio(w, h, maxSize);

    resizeWindow(winname, w, h);
}

void updateNearestCorner(const Point2f refCorner, const Point2f newCorner, Point2f &corner, float &distance) {
    Vec2f diff;
    float dist;
    diff = refCorner - newCorner;
    dist = diff.ddot(diff);

    if (dist < distance) {
        corner = newCorner;
        distance = dist;
    }
}

// porovnani pro serazeni korespondenci od nejlepsi

bool compareDMatch(const DMatch& a, const DMatch &b) {
    return a.distance < b.distance;
}

/**
 * Zajisteni projekce na kouli
 *
 * Pozn. alternativu HomographyBasedEstimator se mi nepodarilo zprovoznit
 */
void projectOnSphere(Mat inputPic, Mat &outputPic, double CamDist, double FocLen, unsigned PicCount) {
    Mat tmp = Mat(inputPic.size(), inputPic.type());

    double alpha = (2 * PI) / (PicCount * inputPic.cols);
    double alphaX, alphaY, angle;

    double cam = (PicCount * inputPic.cols) / (2 * PI);
    Point middle;
    middle.x = (int) (inputPic.cols / 2);
    middle.y = (int) (inputPic.rows / 2);

    //    cerr << middle.x << " " <<middle.y << " " << alpha << endl;

    for (int i = 0; i < inputPic.rows; i++) {

        alphaY = (middle.y - i) * alpha;

        for (int j = 0; j < inputPic.cols; j++) {

            alphaX = (middle.x - j) * alpha;
            angle = sqrt(alphaX * alphaX + alphaY * alphaY);
            //            cerr << alphaX << " " << alphaY << " i: " << i << " j: " << j << " " ;
            cerr.flush();

            double x, y;
            if (angle != 0) {
                x = ((middle.x - j)*((CamDist * sin(angle) * cam) / (CamDist + cam * (1 - cos(angle))))) / (sin(angle) * cam);
                y = ((middle.y - i)*((CamDist * sin(angle) * cam) / (CamDist + cam * (1 - cos(angle))))) / (sin(angle) * cam);
            } else {
                x = 0;
                y = 0;
            }

            Vec3b inputPixel = inputPic.at<Vec3b>(i, j);
            Vec3b outputPixel;
            outputPixel[0] = inputPixel[0];
            outputPixel[1] = inputPixel[1];
            outputPixel[2] = inputPixel[2];
            tmp.at<Vec3b>((int) (middle.y - (int) y), (int) (middle. x - (int) x)) = outputPixel;
        }
    }

    Point2d topLeft(0, 0), bottomRight(inputPic.cols - 1, inputPic.rows - 1);

    int i, j;

    i = topLeft.y;
    j = topLeft.x;

    alphaY = (middle.y - i) * alpha;
    alphaX = (middle.x - j) * alpha;
    angle = sqrt(alphaX * alphaX + alphaY * alphaY);
    if (angle != 0) {
        topLeft.x = ((middle.x - j)*((CamDist * sin(angle) * cam) / (CamDist + cam * (1 - cos(angle))))) / (sin(angle) * cam);
        topLeft.y = ((middle.y - i)*((CamDist * sin(angle) * cam) / (CamDist + cam * (1 - cos(angle))))) / (sin(angle) * cam);
    } else {
        topLeft.x = 0;
        topLeft.y = 0;
    }


    i = bottomRight.y;
    j = bottomRight.x;

    alphaY = (middle.y - i) * alpha;
    alphaX = (middle.x - j) * alpha;
    angle = sqrt(alphaX * alphaX + alphaY * alphaY);
    if (angle != 0) {
        bottomRight.x = ((middle.x - j)*((CamDist * sin(angle) * cam) / (CamDist + cam * (1 - cos(angle))))) / (sin(angle) * cam);
        bottomRight.y = ((middle.y - i)*((CamDist * sin(angle) * cam) / (CamDist + cam * (1 - cos(angle))))) / (sin(angle) * cam);
    } else {
        bottomRight.x = 0;
        bottomRight.y = 0;
    }

    topLeft.x = middle.x - topLeft.x;
    topLeft.y = middle.y - topLeft.y;

    bottomRight.x = middle.x - bottomRight.x;
    bottomRight.y = middle.y - bottomRight.y;

    // Ořezání černých okrajů
    Rect roi(topLeft, bottomRight);

    Mat roiImage = tmp(roi);

    roiImage.copyTo(outputPic);
}

int main(int argc, char* argv[]) {
    // jmena souboru pro zpracovani
    vector<string> imageNames;
    vector<Mat> images;
    vector<Mat> sphereImages;
    string outputName(argv[argc - 1]);

    // zpracovani parametru prikazove radky
    for (int i = 1; i < argc - 1; i++) {
        imageNames.push_back(argv[i]);
    }

    for (int i = 0; i < imageNames.size(); i++) {
        Mat img = imread(imageNames[i]);
        images.push_back(img);
    }

    for (int i = 0; i < images.size(); i++) {
        if (images[i].data == NULL) {
            cerr << "Error: failed to read imput image files!" << i << endl;
            return -1;
        }

        Mat output;
#if VISUAL_DEBUG
        imshowScaled("in", images[i]);
#endif
        projectOnSphere(images[i], output, 8700/*4940*/, 100000, images.size());

        images[i].release();

        sphereImages.push_back(output);

        vector<int> outputParams;
        outputParams.push_back(CV_IMWRITE_JPEG_QUALITY);
#if VISUAL_DEBUG
        imshowScaled("out", output);
        waitKey(0);
        destroyAllWindows();
#endif
    }

    images.clear();

    Mat homography;
    vector<Mat> homographies;

    homographies.push_back(Mat::eye(3, 3, CV_64F));

    for (int i = 0; i < sphereImages.size() - 1; i++) {

        /*************************************/
        //Prevzano z DU: hw04-RANSAC-zadani
        SurfFeatureDetector detector;
        Mat img1 = sphereImages[i];
        Mat img2 = sphereImages[i + 1];

        vector< KeyPoint> keyPoints1, keyPoints2;
        detector.detect(img1, keyPoints1);
        detector.detect(img2, keyPoints2);

        // extraktor SURF descriptoru
        SurfDescriptorExtractor descriptorExtractor;

        // samonty vypocet SURF descriptoru
        Mat descriptors1, descriptors2;
        descriptorExtractor.compute(img1, keyPoints1, descriptors1);
        descriptorExtractor.compute(img2, keyPoints2, descriptors2);

        // tento vektor je pouze pro ucely funkce hledajici korespondence
        vector< Mat> descriptorVector2;
        descriptorVector2.push_back(descriptors2);

        // objekt, ktery dokaze snad pomerne efektivne vyhledavat podebne vektory v prostorech s vysokym poctem dimenzi
        FlannBasedMatcher matcher;
        // Pridani deskriptoru, mezi kterymi budeme pozdeji hledat nejblizsi sousedy
        matcher.add(descriptorVector2);
        // Vytvoreni vyhledavaci struktury nad vlozenymi descriptory
        matcher.train();

        vector<cv::DMatch > matches;
        // Nalezeni korespondenci - nejpodobnejsich descriptoru z obrazku 2 (descriptorVector2)
        // pro oblasti z obrazku 1 (descriptors1).
        matcher.match(descriptors1, matches);

        // serazeni korespondenci od nejlepsi (ma nejmensi vzajemnou vzdalenost v prostoru descriptoru)
        sort(matches.begin(), matches.end(), compareDMatch);
        // pouzijeme jen 200 nejlepsich korespondenci
        matches.resize(min(300, (int) matches.size()));

        // pripraveni korespondujicich dvojic
        Mat img1Pos(matches.size(), 1, CV_32FC2);
        Mat img2Pos(matches.size(), 1, CV_32FC2);

        // naplneni matic pozicemi korespondujicich oblasti
        for (int j = 0; j < (int) matches.size(); j++) {
            img1Pos.at< Vec2f>(j)[0] = keyPoints1[ matches[ j].queryIdx].pt.x;
            img1Pos.at< Vec2f>(j)[1] = keyPoints1[ matches[ j].queryIdx].pt.y;
            img2Pos.at< Vec2f>(j)[0] = keyPoints2[ matches[ j].trainIdx].pt.x;
            img2Pos.at< Vec2f>(j)[1] = keyPoints2[ matches[ j].trainIdx].pt.y;
        }

        // Doplnte vypocet 3x3 matice homografie s vyuzitim algoritmu RANSAC. Pouzijte jdenu funkci knihovny OpenCV.
        homography = findHomography(img1Pos, img2Pos, CV_RANSAC);
        homographies.push_back(homographies.back() * homography);

        /******************************/
    }


    float minX = 0;
    float minY = 0;
    float maxX = (float) sphereImages[0].cols;
    float maxY = (float) sphereImages[0].rows;

    vector< Point2f> projectedCorners;
    for (int i = 0; i < sphereImages.size(); i++) {
        // rohy obrazku 2
        vector< Vec3d> corners;
        corners.push_back(Vec3d(0, 0, 1));
        corners.push_back(Vec3d(sphereImages[i].cols, 0, 1));
        corners.push_back(Vec3d(sphereImages[i].cols, sphereImages[i].rows, 1));
        corners.push_back(Vec3d(0, sphereImages[i].rows, 1));

        homography = homographies[i];

        for (int j = 0; j < (int) corners.size(); j++) {
            Mat projResult = homography.inv() * Mat(corners[j]);

            projectedCorners.push_back(
                    Point2f(
                    (float) (projResult.at<double>(0) / projResult.at<double>(2)),
                    (float) (projResult.at<double>(1) / projResult.at<double>(2))));

            minX = std::min(minX, (float) (projResult.at<double>(0) / projResult.at<double>(2)));
            maxX = std::max(maxX, (float) (projResult.at<double>(0) / projResult.at<double>(2)));
            minY = std::min(minY, (float) (projResult.at<double>(1) / projResult.at<double>(2)));
            maxY = std::max(maxY, (float) (projResult.at<double>(1) / projResult.at<double>(2)));
        }
    }

    //    std::cerr << "minX " << minX << std::endl;
    //    std::cerr << "minY " << minY << std::endl;
    //    std::cerr << "maxX " << maxX << std::endl;
    //    std::cerr << "maxY " << maxY << std::endl;

    Mat perspectiveMatrix = Mat::eye(3, 3, CV_64F);

    // Nalezení krajních bodů
    Point2f topLeft(minX, minY), topRight(maxX, minY), bottomLeft(minX, maxY), bottomRight(maxX, maxY);
    Point2f topLeftNearest, topRightNearest, bottomLeftNearest, bottomRightNearest;
    float topLeftDistance = 1e60, topRightDistance = 1e60, bottomLeftDistance = 1e60, bottomRightDistance = 1e60;
    for (int i = 0; i < projectedCorners.size(); i++) {

        Vec2f corner = projectedCorners[i];

        updateNearestCorner(topLeft, corner, topLeftNearest, topLeftDistance);
        updateNearestCorner(topRight, corner, topRightNearest, topRightDistance);
        updateNearestCorner(bottomLeft, corner, bottomLeftNearest, bottomLeftDistance);
        updateNearestCorner(bottomRight, corner, bottomRightNearest, bottomRightDistance);
    }

    vector<Point2f> srcCorners;
    vector<Point2f> destCorners;

    srcCorners.push_back(Point2f(topLeftNearest));
    srcCorners.push_back(Point2f(topRightNearest));
    srcCorners.push_back(Point2f(bottomLeftNearest));
    srcCorners.push_back(Point2f(bottomRightNearest));

    int width = maxX - minX;
    int height = maxY - minY;
    float scale;

    // Omezené velikosti výsledného obrazu
    scale = maxDimensionsWithRatio(width, height, 1 << 14);

    minX *= scale;
    maxX *= scale;
    minY *= scale;
    maxY *= scale;

    destCorners.push_back(Point2f(0, 0));
    destCorners.push_back(Point2f(width, 0));
    destCorners.push_back(Point2f(0, height));
    destCorners.push_back(Point2f(width, height));

    // Vyrovnání obrazu
    perspectiveMatrix = getPerspectiveTransform(srcCorners, destCorners);

    Mat translateMatrix = Mat::eye(3, 3, CV_64F);
    translateMatrix.at<double>(0, 2) = (double) -minX;
    translateMatrix.at<double>(1, 2) = (double) -minY;

    Mat transformationMatrix = perspectiveMatrix;

    Mat outputBuffer = Mat::zeros(height, width, CV_8UC3); //Vycistim si abych nemel v pozadi binarni smeti.
    for (int i = 0; i < sphereImages.size(); i++) {

        homography = homographies[i];

        warpPerspective(sphereImages[i], outputBuffer, transformationMatrix * homography.inv(), outputBuffer.size(), 1, BORDER_TRANSPARENT);
    }

    Mat finalImg = outputBuffer;

    vector<int> outputParams;
    outputParams.push_back(CV_IMWRITE_JPEG_QUALITY);
    imwrite(outputName, finalImg, outputParams);
#if VISUAL_DEBUG
    imshowScaled("MERGED", finalImg);
    waitKey();
#endif
}
