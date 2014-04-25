/**
Name: Amruta Kajrekar
Project: Panorama Synthesis with image blending and straightening.
Reference: opencv/samples/stitching_detailed class
**/
#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;


/**
The program takes in all the images to be stitched together.
And writes stitched images onto the disk (in the same folder). 
The 2 output images contain stitched images with each feather blending and multiband blending applied to them.
*/
vector<string> allImages;
double work_megapix = 0.6;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
string result_multi = "result_multi.jpg";
string result_feather = "result_feather.jpg";
double work_scale = 1, seam_scale = 1, compose_scale = 1;
bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;
Ptr<FeaturesFinder> featureFinder;
Ptr<WarperCreator> warperCreator;
Ptr<SeamFinder> seamFinder;

int main(int argc, char* argv[])
{
    /** Add all the images to the vector. */
    for (int i = 1; i < argc; ++i) {
       allImages.push_back(argv[i]);
    }

    /** Checking the number of images. */
    int img_no = static_cast<int>(allImages.size());
    if (img_no < 2) {
        LOGLN("Need more images");
        return -1;
    }

    
    featureFinder = new SurfFeaturesFinder();
    Mat full_img, img;
    vector<ImageFeatures> features(img_no);
    vector<Mat> images(img_no);
    vector<Size> full_img_sizes(img_no);
    double seam_work_aspect = 1;

    /** find the features for all the images. */
    for (int i = 0; i < img_no; ++i)
    {
        full_img = imread(allImages[i]);
        full_img_sizes[i] = full_img.size();

        if (full_img.empty())
        {
            return -1;
        }
        if (!is_work_scale_set)
            {
                work_scale = min(1.0, sqrt(0.6 * 1e6 / full_img.size().area()));
                is_work_scale_set = true;
            }
            resize(full_img, img, Size(), work_scale, work_scale);
        
        if (!is_seam_scale_set)
        {
            seam_scale = min(1.0, sqrt(0.1 * 1e6 / full_img.size().area()));
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }

        (*featureFinder)(img, features[i]);
        features[i].img_idx = i;

        resize(full_img, img, Size(), seam_scale, seam_scale);
        images[i] = img.clone();
    }

    featureFinder->collectGarbage();
    full_img.release();
    img.release();

    /** finding the best matching features between the images. */
    vector<MatchesInfo> pairwise_matches;
    BestOf2NearestMatcher matcher(false, 0.3f);
    matcher(features, pairwise_matches);
    matcher.collectGarbage();
    

    vector<int> indices = leaveBiggestComponent(features, pairwise_matches, 1.f);
    vector<Mat> img_subset;
    vector<string> allImages_subset;
    vector<Size> full_img_sizes_subset;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        allImages_subset.push_back(allImages[indices[i]]);
        img_subset.push_back(images[indices[i]]);
        full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
    }

    images = img_subset;
    allImages = allImages_subset;
    full_img_sizes = full_img_sizes_subset;

    img_no = static_cast<int>(allImages.size());
    if (img_no < 2){
        return -1;
    }

    /** Estimate the camera rotation between all the images. */
    HomographyBasedEstimator estimator;
    vector<CameraParams> cameras;
    estimator(features, pairwise_matches, cameras);

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
       
    }

    /** Performing image alignment. */
    Ptr<detail::BundleAdjusterBase> adjuster;
    adjuster = new detail::BundleAdjusterRay();
   
    adjuster->setConfThresh(1.f);
    Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
    for (int k=0;k<2;k++)
    {
        for(int j=0;j<3;j++)
	{
	   refine_mask(k,j) = 1;
	}
    }
    adjuster->setRefinementMask(refine_mask);
    (*adjuster)(features, pairwise_matches, cameras);

    /** Find median focal length. */
    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i) {
       focals.push_back(cameras[i].focal);
    }

    sort(focals.begin(), focals.end());
    float warped_image_scale;
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

   /** Perform image straightening using wave correction (horizontal wave correction) */
    if (true)
    {
        vector<Mat> rmats;
        for (size_t i = 0; i < cameras.size(); ++i)
            rmats.push_back(cameras[i].R);
        waveCorrect(rmats, wave_correct);
        for (size_t i = 0; i < cameras.size(); ++i)
            cameras[i].R = rmats[i];
    }

    vector<Point> corners(img_no);
    vector<Mat> masks_warped(img_no);
    vector<Mat> images_warped(img_no);
    vector<Size> sizes(img_no);
    vector<Mat> masks(img_no);

    /** Preapre masks for each image. */
    for (int i = 0; i < img_no; ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    /** Warp the images and their masks. Using Spherical warping. */
    warperCreator = new cv::SphericalWarper();
 
    if (warperCreator.empty()) {
	return 1;
    }

    Ptr<RotationWarper> warper = warperCreator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

    for (int i = 0; i < img_no; ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect;
        K(0,0) *= swa; K(0,2) *= swa;
        K(1,1) *= swa; K(1,2) *= swa;

        corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();
        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }

    vector<Mat> images_warped_f(img_no);
    for (int i = 0; i < img_no; ++i){
        images_warped[i].convertTo(images_warped_f[i], CV_32F);
    }

    Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
    compensator->feed(corners, images_warped, masks_warped);

    seamFinder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);
    
    if (seamFinder.empty()) {
	return 1;
    }

    seamFinder->find(images_warped_f, corners, masks_warped);

    images.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();

    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;
    Ptr<Blender> blender, blender_multi, blender_feather;
    double compose_work_aspect = 1;

    for (int img_idx = 0; img_idx < img_no; ++img_idx)
    {
        full_img = imread(allImages[img_idx]);
        if (!is_compose_scale_set)
        {
            if (-1 > 0)
                compose_scale = min(1.0, sqrt(-1 * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;
            compose_work_aspect = compose_scale / work_scale;

            warped_image_scale *= static_cast<float>(compose_work_aspect);
            warper = warperCreator->create(warped_image_scale);

            /** Update corners and sizes of each image*/
            for (int i = 0; i < img_no; ++i)
            {
                cameras[i].focal *= compose_work_aspect;
                cameras[i].ppx *= compose_work_aspect;
                cameras[i].ppy *= compose_work_aspect;

                Size sz = full_img_sizes[i];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                    sz.height = cvRound(full_img_sizes[i].height * compose_scale);
                }

                Mat K;
                cameras[i].K().convertTo(K, CV_32F);
                Rect roi = warper->warpRoi(sz, K, cameras[i].R);
                corners[i] = roi.tl();
                sizes[i] = roi.size();
            }
        }
        if (abs(compose_scale - 1) > 1e-1)
            resize(full_img, img, Size(), compose_scale, compose_scale);
        else
            img = full_img;
        full_img.release();
        Size img_size = img.size();

        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);

        warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();

        dilate(masks_warped[img_idx], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size());
        mask_warped = seam_mask & mask_warped;

        /** feather blending */
        if (blender_feather.empty())
        {
	    int blend_type_feather = Blender::FEATHER;
            blender_feather = Blender::createDefault(blend_type_feather, false);

            Size dst_sz = resultRoi(corners, sizes).size();
            float blend_width = sqrt(static_cast<float>(dst_sz.area())) * 5 / 100.f;
            FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender_feather));
            fb->setSharpness(1.f/blend_width);
            blender_feather->prepare(corners, sizes);
        }

	/**  multiband blending */
       if (blender_multi.empty())
        {
	    int blend_type_multi = Blender::MULTI_BAND;
            blender_multi = Blender::createDefault(blend_type_multi , false);
	    
            Size dst_sz = resultRoi(corners, sizes).size();
            float blend_width = sqrt(static_cast<float>(dst_sz.area())) * 5 / 100.f;
            MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender_multi));
            mb->setNumBands(15);
            blender_multi->prepare(corners, sizes);
        }


         /** Blend the current image with feather and multiband blending */
        blender_feather->feed(img_warped_s, mask_warped, corners[img_idx]);
	blender_multi ->feed(img_warped_s, mask_warped, corners[img_idx]);
    }

    Mat result_multi_img, result_feather_img,result_mask;

    blender_feather->blend(result_feather_img, result_mask);
    blender_multi ->blend(result_multi_img, result_mask);

    imwrite(result_multi, result_multi_img);
    imwrite(result_feather, result_feather_img);

    return 0;
}


