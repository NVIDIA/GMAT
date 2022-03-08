#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
// #define N 3

using namespace Eigen;
using namespace std;

typedef Matrix<float, Dynamic, Dynamic, RowMajor>  MatrixXfRowMajor;

static const MatrixXfRowMajor threed_68_points{{-0.7425, -0.3662,  0.4207},
        {-0.7400, -0.1836,  0.5642},
        {-0.6339,  0.0051,  0.1404},
        {-0.5988,  0.1618, -0.0176},
        {-0.5455,  0.3358, -0.0198},
        {-0.4669,  0.4768, -0.1059},
        {-0.3721,  0.5836, -0.1078},
        {-0.2199,  0.6593, -0.3520},
        {-0.0184,  0.7019, -0.4312},
        { 0.1829,  0.6588, -0.4117},
        { 0.3413,  0.5932, -0.2251},
        { 0.4535,  0.5002, -0.1201},
        { 0.5530,  0.3364, -0.0101},
        { 0.6051,  0.1617,  0.0017},
        { 0.6010,  0.0050,  0.2182},
        { 0.7230, -0.1830,  0.5235},
        { 0.7264, -0.3669,  0.3882},
        {-0.5741, -0.5247, -0.1624},
        {-0.4902, -0.6011, -0.3335},
        {-0.3766, -0.6216, -0.4337},
        {-0.2890, -0.6006, -0.4818},
        {-0.1981, -0.5750, -0.5065},
        { 0.1583, -0.5989, -0.5168},
        { 0.2487, -0.6201, -0.4938},
        { 0.3631, -0.6215, -0.4385},
        { 0.4734, -0.6011, -0.3499},
        { 0.5571, -0.5475, -0.1870},
        {-0.0182, -0.3929, -0.5284},
        { 0.0050, -0.2602, -0.6295},
        {-0.0181, -0.1509, -0.7110},
        {-0.0181, -0.0620, -0.7463},
        {-0.1305,  0.0272, -0.5205},
        {-0.0647,  0.0506, -0.5580},
        { 0.0049,  0.0500, -0.5902},
        { 0.0480,  0.0504, -0.5732},
        { 0.1149,  0.0275, -0.5329},
        {-0.4233, -0.3598, -0.2748},
        {-0.3783, -0.4226, -0.3739},
        {-0.2903, -0.4217, -0.3799},
        {-0.2001, -0.3991, -0.3561},
        {-0.2667, -0.3545, -0.3658},
        {-0.3764, -0.3536, -0.3441},
        { 0.1835, -0.3995, -0.3551},
        { 0.2501, -0.4219, -0.3741},
        { 0.3411, -0.4223, -0.3760},
        { 0.4082, -0.3987, -0.3338},
        { 0.3410, -0.3550, -0.3626},
        { 0.2488, -0.3763, -0.3652},
        {-0.2374,  0.2695, -0.4086},
        {-0.1736,  0.2257, -0.5026},
        {-0.0644,  0.1823, -0.5703},
        { 0.0049,  0.2052, -0.5784},
        { 0.0479,  0.1826, -0.5739},
        { 0.1563,  0.2245, -0.5130},
        { 0.2441,  0.2697, -0.4012},
        { 0.1572,  0.3153, -0.4905},
        { 0.0713,  0.3393, -0.5457},
        { 0.0050,  0.3398, -0.5557},
        {-0.0846,  0.3391, -0.5393},
        {-0.1505,  0.3151, -0.4926},
        {-0.2374,  0.2695, -0.4086},
        {-0.0845,  0.2493, -0.5288},
        { 0.0050,  0.2489, -0.5514},
        { 0.0711,  0.2489, -0.5354},
        { 0.2245,  0.2698, -0.4106},
        { 0.0711,  0.2489, -0.5354},
        { 0.0050,  0.2489, -0.5514},
        {-0.0645,  0.2489, -0.5364}};

static void expand_bbox_rectangle(MatrixXfRowMajor &lms, int32_t w, int32_t h, RowVectorX<float> &projected_bbox,
                            float roll=.0f, float expand_forehead=0.3f,
                            float bbox_x_factor=2.0f, float bbox_y_factor=2.0f){
    MatrixXf::Index maxRow, maxCol;
    float min_pt_x = lms(all, 0).minCoeff(&maxRow, &maxCol);
    float max_pt_x = lms(all, 0).maxCoeff(&maxRow, &maxCol);
    float min_pt_y = lms(all, 1).minCoeff(&maxRow, &maxCol);
    float max_pt_y = lms(all, 1).maxCoeff(&maxRow, &maxCol);

    int32_t bbox_size_x = static_cast<int>(((max_pt_x - min_pt_x) * bbox_x_factor));
    float center_pt_x = 0.5 * min_pt_x + 0.5 * max_pt_x;

    int32_t bbox_size_y = static_cast<int>(((max_pt_y - min_pt_y) * bbox_y_factor));
    float center_pt_y = 0.5 * min_pt_y + 0.5 * max_pt_y;

    float bbox_min_x = center_pt_x - bbox_size_x * 0.5, bbox_max_x = center_pt_x + bbox_size_x * 0.5;
    float bbox_min_y = center_pt_y - bbox_size_y * 0.5, bbox_max_y = center_pt_y + bbox_size_y * 0.5;

    float expand_forehead_size = .0f;
    if (abs(roll) > 2.5){
        expand_forehead_size = expand_forehead * (max_pt_y - min_pt_y);
        bbox_max_y += expand_forehead_size;
    }
    else if(roll > 1){
        expand_forehead_size = expand_forehead * (max_pt_x - min_pt_x);
        bbox_max_x += expand_forehead_size;
    }
    else if (roll < -1){
        expand_forehead_size = expand_forehead * (max_pt_x - min_pt_x);
        bbox_min_x -= expand_forehead_size;
    }
    else{
        expand_forehead_size = expand_forehead * (max_pt_y - min_pt_y);
        bbox_min_y -= expand_forehead_size;
    }

    int32_t bbox_min_x_int = static_cast<int32_t>(bbox_min_x);
    int32_t bbox_max_x_int = static_cast<int32_t>(bbox_max_x);
    int32_t bbox_min_y_int = static_cast<int32_t>(bbox_min_y);
    int32_t bbox_max_y_int = static_cast<int32_t>(bbox_max_y);

    int32_t padding_left = abs(min(bbox_min_x_int, 0));
    int32_t padding_top = abs(min(bbox_min_y_int, 0));
    int32_t padding_right = max(bbox_max_x_int - w, 0);
    int32_t padding_bottom = max(bbox_max_y_int - h, 0);

    float crop_left = padding_left > 0 ? 0 : bbox_min_x_int;
    float crop_top = padding_top > 0 ? 0 : bbox_min_y_int;
    float crop_right = padding_right > 0 ? w : bbox_max_x_int;
    float crop_bottom = padding_bottom > 0 ? h : bbox_max_y_int;

    projected_bbox = RowVectorX<float>{{crop_left, crop_top, crop_right, crop_bottom}};
}

static void plot_3d_landmark(const MatrixXfRowMajor &points, VectorXf &campose, const MatrixXfRowMajor &intrinsics,
                    MatrixXfRowMajor &lms){
    auto rvec = campose(seq(0, 2));
    auto tvec = campose(seq(3, 5));
    MatrixXfRowMajor rmat_tr = AngleAxis<float>(rvec.norm(), rvec / rvec.norm()).toRotationMatrix().transpose();
    // cout << "rmat_tr: \n" << rmat_tr << endl;
    MatrixXfRowMajor lm_3d_trans = (points * rmat_tr).rowwise() + tvec.transpose();
    // cout << "rmat_tr + tvec: \n" << rmat_tr.colwise() + tvec << endl;
    // cout << "lm_3d_trans: \n" << lm_3d_trans << endl;

    MatrixXfRowMajor lms_3d_trans_proj = (intrinsics * lm_3d_trans.transpose()).transpose();
    MatrixXfRowMajor proj(2, lm_3d_trans.rows());
    proj << lms_3d_trans_proj(all, 2).transpose(), lms_3d_trans_proj(all, 2).transpose();
    // cout << "proj:\n" << proj << endl;
    lms = lms_3d_trans_proj(all, seq(0, 1)).array() / proj.transpose().array();
    // cout << "lms_projected: \n" << lms_projected << endl;
}

static void pose_bbox_to_full_image(VectorXf &pose, const MatrixXfRowMajor &image_intrinsics,
                            VectorXf &bbox, VectorXf &global_dof){
    VectorXf tvec = pose(seq(3, 5));
    VectorXf rvec = pose(seq(0,2));

    float bbox_center_x = bbox(0) + static_cast<int32_t>(((bbox(2) - bbox(0)) / 2));
    float bbox_center_y = bbox(1) + static_cast<int32_t>((bbox(3) - bbox(1)) / 2);
    // cout << "bbox_centerx: " << bbox_center_x << endl;

    MatrixXfRowMajor bbox_intrinsics = image_intrinsics;
    bbox_intrinsics(0, 2) = bbox_center_x;
    bbox_intrinsics(1, 2) = bbox_center_y;
    // cout << "bbox_intrinsics: \n" << bbox_intrinsics << endl;

    float focal_length = image_intrinsics(0, 0);
    float bbox_width = bbox[2] - bbox[0];
    float bbox_height = bbox[3] - bbox[1];
    float bbox_size = bbox_width + bbox_height;

    tvec(2) *= focal_length / bbox_size;
    // cout << "tvec: " << tvec << endl;
    auto projected_point = bbox_intrinsics * tvec;
    // cout << "projected_point: \n" << projected_point << endl;
    tvec = image_intrinsics.transpose().inverse().transpose() * projected_point;
    // cout << "image_intrinsics.transpose().inverse(): \n" << image_intrinsics.transpose().inverse() << endl;
    // cout << "tvec: " << tvec << endl;

    auto rmat = AngleAxis<float>(rvec.norm(), rvec / rvec.norm()).toRotationMatrix();
    // cout << "rvec / norm(): \n" << rvec/rvec.norm() << endl;
    auto projected_point_r = bbox_intrinsics * rmat;
    rmat = image_intrinsics.transpose().inverse().transpose() * projected_point_r;
    // cout << "rmat: \n" << rmat << endl;
    AngleAxis<float> rvec_(rmat);
    rvec = rvec_.axis() * rvec_.angle();
    // cout << "rvec :\n" << rvec << endl;

    global_dof << rvec, tvec;
}

static void transform_pose_global_project_bbox(
    Map<MatrixXfRowMajor> &boxes,
    Map<MatrixXfRowMajor> &dofs,
    VectorXf &pose_mean,
    VectorXf &pose_stddev,
    const array<int64_t, 3> &image_shape,
    const MatrixXfRowMajor &tnreed_68_points,
    MatrixXfRowMajor &global_dofs,
    MatrixXfRowMajor &projected_boxes,
    float bbox_x_factor=1.1f,
    float bbox_y_factor=1.1f,
    float expand_forehead=0.3f) {

    if (boxes.rows() == 0) return;

    float width = image_shape[2], height = image_shape[1];
    const MatrixXfRowMajor global_intrinsics {
        {width + height, .0f,              width / 2.0f },
        {.0f,              width + height, height / 2.0f},
        {.0f,              .0f,              1.0f         }
    };
    // cout << "dofs: \n" << dofs << endl;
    // cout << "pose_stddev: \n" << pose_stddev << endl;
    // MatrixXf dofs_stat = (dofs.array().colwise()) * pose_stddev.array();
    MatrixXfRowMajor dofs_stat = (dofs * pose_stddev.asDiagonal());
    dofs_stat.rowwise() += pose_mean.transpose();
    // cout << "dofs_stat: \n" << dofs_stat << endl;
    // cout << "boxes: \n" << boxes << endl;
    global_dofs = MatrixXfRowMajor(dofs_stat.rows(), 6);
    projected_boxes = MatrixXfRowMajor(dofs_stat.rows(), 4);
    for (int i = 0; i < dofs_stat.rows(); i++){
        auto dof = static_cast<VectorXf>(dofs_stat.row(i));
        auto box = static_cast<VectorXf>(boxes.row(i));
        // auto dof_vec = static_cast<VectorXf>(dof);
        VectorXf global_dof(6);
        pose_bbox_to_full_image(dof, global_intrinsics, box, global_dof);
        // cout << "global_dof: \n" << global_dof << endl;

        MatrixXfRowMajor lms;
        plot_3d_landmark(threed_68_points, global_dof, global_intrinsics, lms);

        RowVectorX<float> projected_bbox;
        expand_bbox_rectangle(lms, width, height, projected_bbox, global_dof(2), expand_forehead, bbox_x_factor, bbox_y_factor);
        // cout << "projected_bbox:" << projected_bbox << endl;

        // global_dofs << global_dof.transpose();
        // projected_boxes << projected_bbox;
        global_dofs.row(i) = global_dof;
        projected_boxes.row(i) = projected_bbox;
    }
    // cout << "global_dofs: \n" << global_dofs << endl;
    // cout << "projected_boxes: \n" << projected_boxes << endl;

}
