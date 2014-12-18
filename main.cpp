
// Muzete si vypnout vykreslovani na obrazovku.
#define VISUAL_OUTPUT 1

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

// porovnani pro serazeni korespondenci od nejlepsi
bool compareDMatch( const DMatch& a, const DMatch &b){
    return a.distance < b.distance;
}

void projectOnSphere(Mat inputPic, Mat outputPic, double CamDist, double FocLen, unsigned PicCount){
//    outputPic = inputPic.clone();
//    outputPic = Mat::zeros(outputPic.size(), outputPic.type());

    double alpha = (2*PI) / (PicCount * inputPic.cols);

    double cam = (PicCount*inputPic.cols)/(2*PI);
    Point middle;
    middle.x = (int)(inputPic.cols/2);
    middle.y = (int)(inputPic.rows/2);

    cerr << middle.x << " " <<middle.y << " " << alpha << endl;

    for(int i = 0; i < inputPic.rows; i++){

        double alphaY = (middle.y - i) * alpha;

        for(int j = 0; j < inputPic.cols; j++){

            double alphaX = (middle.x - j) * alpha;
            double angle = sqrt(alphaX * alphaX + alphaY * alphaY);
            //            cerr << alphaX << " " << alphaY << " i: " << i << " j: " << j << " " ;
            cerr.flush();

            double x,y;
            if(angle != 0){
                x = ((middle.x - j)*((CamDist * sin(angle) * cam)/(CamDist + cam*(1 - cos(angle)))))/(sin(angle) * cam);
                y = ((middle.y - i)*((CamDist * sin(angle) * cam)/(CamDist + cam*(1 - cos(angle)))))/(sin(angle) * cam);
            }
            else{
                x = 0;
                y = 0;
            }
//            cerr <<  "x: " << x<< " " << y << " " << endl;
            cerr.flush();

            outputPic.at<Vec3b>((int)(middle.y - (int)y), (int)(middle. x - (int)x)) = inputPic.at<Vec3b>(i,j);
        }
        //        cerr << endl;
    }
}

int main( int argc, char* argv[])
{
    // jmena souboru pro zpracovani
    vector<string> imageNames;
    vector<Mat> images;
    vector<Mat> sphereImages;

    // zpracovani parametru prikazove radky
    for( int i = 1; i < argc; i++){
        imageNames.push_back(argv[i]);
    }

    for(int i = 0; i < imageNames.size(); i++){
        Mat img = imread(imageNames[i]);
        images.push_back(img);
    }

    for(int i = 0; i < images.size(); i++){
        if(images[i].data == NULL){
            cerr << "Error: failed to read imput image files!" << i << endl;
            return -1;
        }

        Mat output = Mat::zeros(images[i].size(),images[i].type());
        cerr << output.size() << endl;
//        imshow("in", images[i]);
        projectOnSphere(images[i],output, 9000/*4940*/, 100000, images.size());

        sphereImages.push_back(output);

        vector<int> outputParams;
        outputParams.push_back(CV_IMWRITE_JPEG_QUALITY);
//        imshow("out", output);
        string m= to_string(i);
        imshow(m.c_str(),output);
    }

//    Mat outPut;

    vector<Mat> homographys;

    for(int i = 0; i < sphereImages.size() - 1; i++){
        SurfFeatureDetector detector;
        Mat img1 = sphereImages[i];
        Mat img2 = sphereImages[i + 1];

        vector< KeyPoint> keyPoints1, keyPoints2;
        detector.detect( img1, keyPoints1);
        detector.detect( img2, keyPoints2);


        // extraktor SURF descriptoru
        SurfDescriptorExtractor descriptorExtractor;

        // samonty vypocet SURF descriptoru
        Mat descriptors1, descriptors2;
        descriptorExtractor.compute( img1, keyPoints1, descriptors1);
        descriptorExtractor.compute( img2, keyPoints2, descriptors2);

        // tento vektor je pouze pro ucely funkce hledajici korespondence
        vector< Mat> descriptorVector2;
        descriptorVector2.push_back( descriptors2);

        // objekt, ktery dokaze snad pomerne efektivne vyhledavat podebne vektory v prostorech s vysokym poctem dimenzi
        FlannBasedMatcher matcher;
        // Pridani deskriptoru, mezi kterymi budeme pozdeji hledat nejblizsi sousedy
        matcher.add( descriptorVector2);
        // Vytvoreni vyhledavaci struktury nad vlozenymi descriptory
        matcher.train();

        vector<cv::DMatch > matches;
        // Doplòte nalezeni korespondenci - nejpodobnejsich descriptoru z obrazku 2 (descriptorVector2)
        // pro oblasti z obrazku 1 (descriptors1). Vysledek ulozte do matches.
        matcher.match(descriptors1, matches);

        // serazeni korespondenci od nejlepsi (ma nejmensi vzajemnou vzdalenost v prostoru descriptoru)
        sort( matches.begin(), matches.end(), compareDMatch);
        // pouzijeme jen 200 nejlepsich korespondenci
        matches.resize( min( 500, (int) matches.size()));

        // pripraveni korespondujicich dvojic
        Mat img1Pos( matches.size(), 1, CV_32FC2);
        Mat img2Pos( matches.size(), 1, CV_32FC2);

        // naplneni matic pozicemi korespondujicich oblasti
        for( int j = 0; j < (int)matches.size(); j++){
            img1Pos.at< Vec2f>( j)[0] = keyPoints1[ matches[ j].queryIdx].pt.x;
            img1Pos.at< Vec2f>( j)[1] = keyPoints1[ matches[ j].queryIdx].pt.y;
            img2Pos.at< Vec2f>( j)[0] = keyPoints2[ matches[ j].trainIdx].pt.x;
            img2Pos.at< Vec2f>( j)[1] = keyPoints2[ matches[ j].trainIdx].pt.y;
        }

        // Doplnte vypocet 3x3 matice homografie s vyuzitim algoritmu RANSAC. Pouzijte jdenu funkci knihovny OpenCV.
        homographys.push_back(findHomography(img1Pos, img2Pos, CV_RANSAC));
    }

    float minX = 0;
    float minY = 0;
    float maxX = (float) sphereImages[0].cols;
    float maxY = (float) sphereImages[0].rows;

    Mat homography;
    homographys[0].copyTo(homography);

    for(int i = 0; i < sphereImages.size(); i++){
        // rohy obrazku 2
        vector< Vec3d> corners;
        corners.push_back( Vec3d( 0, 0, 1));
        corners.push_back( Vec3d( sphereImages[i].cols, 0, 1));
        corners.push_back( Vec3d( sphereImages[i].cols, sphereImages[i].rows, 1));
        corners.push_back( Vec3d( 0, sphereImages[i].rows, 1));

        if(i > 1){
            homography = homographys[i - 1] * homography;

            for( int j = 0; j < (int)corners.size(); j++){
                // Doplnte promitnuti Mat( corners[ i]) do prostoru obrazku 1 pomoci homography.
                // Dejte si pozor odkud kam homography je. Podle toho pouzijte homography, nebo homography.inv().
                Mat projResult = homography.inv() * Mat( corners[j]);

                minX = std::min( minX, (float) (projResult.at<double>( 0) / projResult.at<double>( 2)));
                maxX = std::max( maxX, (float) (projResult.at<double>( 0) / projResult.at<double>( 2)));
                minY = std::min( minY, (float) (projResult.at<double>( 1) / projResult.at<double>( 2)));
                maxY = std::max( maxY, (float) (projResult.at<double>( 1) / projResult.at<double>( 2)));
            }

        }
    }

    std::cerr << "minX " << minX << std::endl;
    std::cerr << "minY " << minY << std::endl;
    std::cerr << "maxX " << maxX << std::endl;
    std::cerr << "maxY " << maxY << std::endl;

    vector<Mat> outputs;
    for(int i = 0; i < sphereImages.size();i++){
    Mat outputBuffer = Mat::zeros( maxY - minY, maxX - minX, CV_8UC3); //Vycistim si abych nemel v pozadi binarni smeti.
    outputs.push_back(outputBuffer);
    }
    Mat translateMatrix = Mat::eye( 3, 3, CV_64F);

    homographys[0].copyTo(homography);


    minX = 0;
    minY = 0;
    maxX = (float) sphereImages[0].cols;
    maxY = (float) sphereImages[0].rows;

    for(int i = 0; i < sphereImages.size(); i++){
        // rohy obrazku 2
        vector< Vec3d> corners;
        corners.push_back( Vec3d( 0, 0, 1));
        corners.push_back( Vec3d( sphereImages[i].cols, 0, 1));
        corners.push_back( Vec3d( sphereImages[i].cols, sphereImages[i].rows, 1));
        corners.push_back( Vec3d( 0, sphereImages[i].rows, 1));

        if(i > 1){
            homography = homographys[i - 1] * homography;

            std::cerr << "homografie " << std::endl;
            for(int i = 0; i < 3 ; i++){
                for(int j = 0; j < 3; j++){
                    std::cerr << homography.at<float>(Point(i,j)) << " ";
                }
                cerr << endl;
            }
        }

        for( int j = 0; j < (int)corners.size(); j++){
            Mat projResult = homography.inv() * Mat( corners[j]);

            minX = std::min( minX, (float) (projResult.at<double>( 0) / projResult.at<double>( 2)));
            maxX = std::max( maxX, (float) (projResult.at<double>( 0) / projResult.at<double>( 2)));
            minY = std::min( minY, (float) (projResult.at<double>( 1) / projResult.at<double>( 2)));
            maxY = std::max( maxY, (float) (projResult.at<double>( 1) / projResult.at<double>( 2)));
        }

        translateMatrix.at<double>(0,2) = (double)-minX;
        translateMatrix.at<double>(1,2) = (double)-minY;
    }

    cerr << translateMatrix.at<double>(0,2) << " " << translateMatrix.at<double>(1,2) << endl;

    homographys[0].copyTo(homography);

    for(int i = 0; i < sphereImages.size(); i++){

        if(i > 1){
            homography = homographys[i - 1] * homography;

            std::cerr << "homografie " << std::endl;
            for(int i = 0; i < 3 ; i++){
                for(int j = 0; j < 3; j++){
                    std::cerr << homography.at<float>(Point(i,j)) << " ";
                }
                cerr << endl;
            }
        }

        if(i > 0)
            warpPerspective( sphereImages[i], outputs[i], translateMatrix * homography.inv(), outputs[i].size(), 1, BORDER_TRANSPARENT);
        else
            warpPerspective( sphereImages[i], outputs[i], translateMatrix, outputs[i].size(), 1, BORDER_TRANSPARENT);

    }
        Mat final = outputs[0];

        std::cerr << outputs.size() << std::endl;
        for(int i = 1; i < outputs.size(); i++){
            for(int j = 0; j < outputs[i].rows;j++){
                for(int k = 0; k < outputs[i].cols; k++){
                    if((outputs[i].at<Vec3b>(j,k)[0] || outputs[i].at<Vec3b>(j,k)[1]|| outputs[i].at<Vec3b>(j,k)[2]) &&
                       (!final.at<Vec3b>(j,k)[0] && !final.at<Vec3b>(j,k)[1] && !final.at<Vec3b>(j,k)[2]) ){
                        final.at<Vec3b>(j,k) = outputs[i].at<Vec3b>(j,k);
                    }
                }
            }
        }

        vector<int> outputParams;
        outputParams.push_back(CV_IMWRITE_JPEG_QUALITY);
        imwrite("output.jpeg", final,outputParams);
        imshow( "MERGED", final);
        waitKey();

/*

    // SURF detektor lokalnich oblasti
    SurfFeatureDetector detector;

    // samotna detekce lokalnich priznaku
    vector< KeyPoint> keyPoints1, keyPoints2;
    detector.detect( img1, keyPoints1);
    detector.detect( img2, keyPoints2);
    cout << keyPoints1.size() << " " << keyPoints2.size();

    // extraktor SURF descriptoru
    SurfDescriptorExtractor descriptorExtractor;

    // samonty vypocet SURF descriptoru
    Mat descriptors1, descriptors2;
    descriptorExtractor.compute( img1, keyPoints1, descriptors1);
    descriptorExtractor.compute( img2, keyPoints2, descriptors2);

    // tento vektor je pouze pro ucely funkce hledajici korespondence
    vector< Mat> descriptorVector2;
    descriptorVector2.push_back( descriptors2);

    // objekt, ktery dokaze snad pomerne efektivne vyhledavat podebne vektory v prostorech s vysokym poctem dimenzi
    FlannBasedMatcher matcher;
    // Pridani deskriptoru, mezi kterymi budeme pozdeji hledat nejblizsi sousedy
    matcher.add( descriptorVector2);
    // Vytvoreni vyhledavaci struktury nad vlozenymi descriptory
    matcher.train();

    vector<cv::DMatch > matches;
    // Doplòte nalezeni korespondenci - nejpodobnejsich descriptoru z obrazku 2 (descriptorVector2)
    // pro oblasti z obrazku 1 (descriptors1). Vysledek ulozte do matches.
    matcher.match(descriptors1, matches);

    // serazeni korespondenci od nejlepsi (ma nejmensi vzajemnou vzdalenost v prostoru descriptoru)
    sort( matches.begin(), matches.end(), compareDMatch);
    // pouzijeme jen 200 nejlepsich korespondenci
    matches.resize( min( 400, (int) matches.size()));

    // pripraveni korespondujicich dvojic
    Mat img1Pos( matches.size(), 1, CV_32FC2);
    Mat img2Pos( matches.size(), 1, CV_32FC2);

    // naplneni matic pozicemi korespondujicich oblasti
    for( int i = 0; i < (int)matches.size(); i++){
        img1Pos.at< Vec2f>( i)[0] = keyPoints1[ matches[ i].queryIdx].pt.x;
        img1Pos.at< Vec2f>( i)[1] = keyPoints1[ matches[ i].queryIdx].pt.y;
        img2Pos.at< Vec2f>( i)[0] = keyPoints2[ matches[ i].trainIdx].pt.x;
        img2Pos.at< Vec2f>( i)[1] = keyPoints2[ matches[ i].trainIdx].pt.y;
    }

    // Doplnte vypocet 3x3 matice homografie s vyuzitim algoritmu RANSAC. Pouzijte jdenu funkci knihovny OpenCV.
    Mat homography = findHomography(img1Pos, img2Pos, CV_RANSAC);


    // vystupni buffer pro vykresleni spojenych obrazku
    Mat outputBuffer = Mat::zeros( 1024, 1280, CV_8UC1); //Vycistim si abych nemel v pozadi binarni smeti.


    // Vysledny spojeny obraz budeme chtit vykreslit do outputBuffer tak, aby se dotykal okraju, ale nepresahoval je.
    // "Prilepime" obrazek 2 k prvnimu. Tuto "slepeninu" je potreba zvetsit a posunout, aby byla na pozadovane pozici.
    // K tomuto potrebujeme zjistit maximalni a minimalni souradnice vykreslenych obrazu. U obrazu 1 je to jednoduche, minima a maxima se
    // ziskaji primo z rozmeru obrazu. U obrazku 2 musime pomoci drive ziskane homografie promitnout do prostoru obrazku 1 jeho rohove body.

    float minX = 0;
    float minY = 0;
    float maxX = (float) img1.cols;
    float maxY = (float) img1.rows;

    // rohy obrazku 2
    vector< Vec3d> corners;
    corners.push_back( Vec3d( 0, 0, 1));
    corners.push_back( Vec3d( img2.cols, 0, 1));
    corners.push_back( Vec3d( img2.cols, img2.rows, 1));
    corners.push_back( Vec3d( 0, img2.rows, 1));

    // promitnuti rohu obrazku 2 do prosotoru obrazku 1 a upraveni minim a maxim
    for( int i = 0; i < (int)corners.size();i ++){

        // Doplnte promitnuti Mat( corners[ i]) do prostoru obrazku 1 pomoci homography.
        // Dejte si pozor odkud kam homography je. Podle toho pouzijte homography, nebo homography.inv().
        Mat projResult = homography.inv() * Mat( corners[i]);

        minX = std::min( minX, (float) (projResult.at<double>( 0) / projResult.at<double>( 2)));
        maxX = std::max( maxX, (float) (projResult.at<double>( 0) / projResult.at<double>( 2)));
        minY = std::min( minY, (float) (projResult.at<double>( 1) / projResult.at<double>( 2)));
        maxY = std::max( maxY, (float) (projResult.at<double>( 1) / projResult.at<double>( 2)));
    }




    // Posuneme a zvetseme/zmenseme vysledny spojeny obrazek tak, by vysledny byl co nejvetsi, ale aby byl uvnitr vystupniho bufferu.

    // Zmena velikosti musi byt takova, aby se nam vysledek vesel na vysku i na sirku
    double scaleFactor = min( outputBuffer.cols / ( maxX - minX), outputBuffer.rows / ( maxY - minY));

    // Doplnte pripraveni matice, ktera zmeni velikost o scaleFactor (vynasobeni timto faktorem) a posune vysledek o -minX a -minY.
    // Po tomto bude obrazek uprostred.
    Mat scaleMatrix = Mat::eye( 3, 3, CV_64F);
    Mat translateMatrix = Mat::eye( 3, 3, CV_64F);


    scaleMatrix.at<double>(0,0) = scaleFactor;
    scaleMatrix.at<double>(1,1) = scaleFactor;

    translateMatrix.at<double>(0,2) = (double)-minX;
    translateMatrix.at<double>(1,2) = (double)-minY;

    Mat centerMatrix = scaleMatrix * translateMatrix;


    // Transformace obrazku 1
    warpPerspective( img1, outputBuffer, centerMatrix, outputBuffer.size(), 1, BORDER_TRANSPARENT);

    // Transformace obrazku 2
    warpPerspective( img2, outputBuffer, centerMatrix * homography.inv(), outputBuffer.size(), 1, BORDER_TRANSPARENT);

    cout << "normMatrix" << endl;
    cout << centerMatrix << endl << endl;

    cout << "normMatrix" << endl;
    cout << homography << endl << endl;

    imshow( "IMG1", img1);
    imshow( "IMG2", img2);
    imshow( "MERGED", outputBuffer);
    waitKey();*/
}



//Ptr<WarperCreator> warper_creator = new cv::CylindricalWarper();

//Ptr<detail::RotationWarper> m = warper_creator->create(1);
//Mat CamIntrinsicParams  = Mat::zeros(3,3,CV_32F);
////        CamIntrinsicParams = Mat::zeros(3,3,CV_32F);
////        CamIntrinsicParams.diag(1);
//float radius = 2228;
//CamIntrinsicParams.at<float>(Point(0,0)) = 400 * 70;
//CamIntrinsicParams.at<float>(Point(0,1)) = 0;
//CamIntrinsicParams.at<float>(Point(0,2)) = (images[i].cols)/2;
//CamIntrinsicParams.at<float>(Point(1,0)) = 0;
//CamIntrinsicParams.at<float>(Point(1,1)) = 400 * 70;
//CamIntrinsicParams.at<float>(Point(1,2)) = images[i].rows/2;
//CamIntrinsicParams.at<float>(Point(2,0)) = 0;
//CamIntrinsicParams.at<float>(Point(2,1)) = 0;
//CamIntrinsicParams.at<float>(Point(2,2)) = 1;

//Mat rotMat = Mat::eye(3,3,CV_32F);

//for(int j = 0; j < 3 ; j++){
//    for(int k = 0; k < 3; k++){
////                std::cerr << j << " " << k << " ";
//        std::cerr << CamIntrinsicParams.at<float>(Point(j,k)) << " ";
//    }
//    std::cerr << std::endl;
//}

//std::cerr << std::endl;

//for(int j = 0; j < 3 ; j++){
//    for(int k = 0; k < 3; k++){
////                std::cerr << j << " " << k << " ";
//        std::cerr << rotMat.at<float>(Point(j,k)) << " ";
//    }
//    std::cerr << std::endl;
//}

////        m = m.create(5);
//Mat output = Mat::zeros(images[i].rows, images[i].cols, CV_32F);
//m->warp(images[i],CamIntrinsicParams,rotMat,INTER_LINEAR, BORDER_REFLECT, output);
//imshow("vystup:", output);
