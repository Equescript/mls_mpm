#ifdef _WIN32
#define EXPORT extern "C" __declspec( dllexport )
#else
#define EXPORT extern "C" __attribute__((visibility ("default")))
#endif

#include "eigen_type.h"
#include "field.h"
#include <iostream>
#include <math.h>
#include <Eigen/Eigen>

template<typename T>
using Array = Field<T>; // one dimension field

class Particle {
public:
    double mass; // particle mass
    Array<Vec2> x; // position
    Array<Vec2> v; // velocity
    Array<Mat2> C; // affine velocity field
    Array<Mat2> F; // deformation gradient
    Array<double> J; // density
    Array<int> material;
    Particle() {}
    Particle(double mass, Field<Vec2> x, Field<Vec2> v, Field<Mat2> C, Field<Mat2> F, Field<double> J, Field<int> material):
    mass(mass), x(x), v(v), C(C), F(F), J(J), material(material) {}
};

class Grid {
public:
    double dx;
    double inv_dx;
    double inv_dx_square;
    Field<Vec2> v; // velocity
    Field<double> m; // grid mass
    Grid() {}
    Grid(double dx, Field<Vec2> v, Field<double> m): dx(dx), inv_dx(1.0/dx), inv_dx_square(1.0/(dx*dx)), v(v), m(m) {}
};

class Scene {
public:
    double dt;
    Particle P;
    Grid G;
    Scene() {}
    Scene(double dt, Particle P, Grid G): dt(dt), P(P), G(G) {}
};

// const int _n_particles = 8192;
// const int _n_particles = 2048;
// quality = 1  # Use a larger value for higher-res simulations
// n_particles, n_grid = 9000 * quality**2, 128 * quality
const int quality = 1;
// const int _n_particles = 9000 * quality * quality;
const int _n_grid = 128 * quality;
const double _dx = 1.0 / _n_grid;
const double _dt = 2e-4;
const double _p_rho = 1.0;
const double _p_vol = (_dx * 0.5) * (_dx * 0.5);
const double _p_mass = _p_vol * _p_rho;
const double _gravity = 9.8;
const int _bound = 3;
const double _E = 400.0; // Young's modulus
const double _nu = 0.2; // Poisson's ratio
const double _mu_0 = _E / (2.0 * (1.0 + _nu));
const double _lambda_0 = _E * _nu / ((1.0 + _nu) * (1.0 - 2.0 * _nu));

Scene SCENE;

Vec2 G_v_data[_n_grid * _n_grid];
double G_m_data[_n_grid * _n_grid];

Vec2 clamp(Vec2 v, double min, double max) {
    v = (min < v.array()).select(v, min);
    v = (max > v.array()).select(v, max);
    return v;
}

double clamp(double v, double min, double max) {
    if (v < min) {
        v = min;
    } else if (v > max) {
        v = max;
    }
    return v;
}

EXPORT void init(Vec2* P_x_data, int* P_material_data, int _n_particles) {
    SCENE = Scene(_dt, Particle(
        _p_mass,
        Field<Vec2>(P_x_data, _n_particles),
        Field<Vec2>(new Vec2[_n_particles], _n_particles),
        Field<Mat2>(new Mat2[_n_particles], _n_particles),
        Field<Mat2>(new Mat2[_n_particles], _n_particles),
        Field<double>(new double[_n_particles], _n_particles),
        Field<int>(P_material_data, _n_particles)
    ), Grid(
        _dx,
        Field<Vec2>(G_v_data, _n_grid, _n_grid),
        Field<double>(G_m_data, _n_grid, _n_grid)
    ));
    for (int i=0; i<_n_particles; i++) {
        SCENE.P.v[i] = Vec2::Zero();
        SCENE.P.C[i] = Mat2::Zero();
        SCENE.P.F[i] = Mat2::Identity();
        SCENE.P.J[i] = 1;
    }
}

void calculate_weight_parameters(const Field<Vec2> &P_x, double dx, int p, Vec2 w[3], Vec2i &origin, Vec2 &x_to_origin) {
    Vec2 x_in_G = P_x[p] / dx;
    Vec2i grid_index = x_in_G.array().floor().cast<int>();
    Vec2 grid_center = grid_index.cast<double>() + Vec2(0.5, 0.5);
    Vec2 x_to_center = x_in_G - grid_center;
    Vec2 w_0 = (x_to_center - Vec2(0.5, 0.5)).array().abs2();
    Vec2 w_1 = x_to_center.array().abs2();
    Vec2 w_2 = (x_to_center + Vec2(0.5, 0.5)).array().abs2();
    w[0] = 0.5 * w_0;
    w[1] = Vec2(0.75, 0.75) - w_1;
    w[2] = 0.5 * w_2;
    origin = grid_index - Vec2i(1, 1);
    x_to_origin = x_to_center + Vec2(1.0, 1.0); // x relative to origin
}

Mat2 calculate_affine_matrix(Scene &scene, int p) {
    // F[p]: deformation gradient update
    scene.P.F[p] = (Mat2::Identity() + scene.dt * scene.P.C[p]) * scene.P.F[p];
    double h = clamp(exp(10.0 * (1.0 - scene.P.J[p])), 0.1, 5);
    double mu = _mu_0 * h;
    double la = _lambda_0 * h;
    if (scene.P.material[p] == 0) { // liquid
        mu = 0.0;
    }
    Eigen::JacobiSVD<Mat2> svd(scene.P.F[p], Eigen::ComputeThinU|Eigen::ComputeThinV);
    Vec2 sig = svd.singularValues();
    // Avoid zero eigenvalues because of numerical errors
    double min = 1e-6;
    sig = (sig.array() > min).select(sig, min);
    Mat2 U = svd.matrixU();
    Mat2 V = svd.matrixV();
    double J = 1.0; // For the liquid, it's always 1.0
    for (int d=0; d<2; d++) {
        double new_sig = sig[d];
        if (scene.P.material[p] == 2) {
            new_sig = clamp(sig[d], 1.0 - 2.5e-2, 1.0 + 4.5e-3);
            scene.P.J[p] *= sig[d] / new_sig;
            sig[d] = new_sig;
        }
        J *= new_sig;
    }
    if (scene.P.material[p] == 0) {
        scene.P.F[p] = Mat2::Identity() * sqrt(J);
    } else if (scene.P.material[p] == 2) {
        scene.P.F[p] = U * sig.asDiagonal() * V.transpose();
    }
    Mat2 stress = 2.0 * mu * (scene.P.F[p] - U * V.transpose()) * scene.P.F[p].transpose() + Mat2::Identity() * la * J * (J - 1.0);
    stress = (-scene.dt * _p_vol * 4 * scene.G.inv_dx_square) * stress;
    return stress + scene.P.mass * scene.P.C[p];
}

void boundary(Field<Vec2> &v, int bound, Vec2i &pos) {
    if (pos.x() < bound && v.get(pos).x() < 0) {
        v.get(pos).x() = 0;
    } else if (pos.x() > (v.row - bound) && v.get(pos).x() > 0) {
        v.get(pos).x() = 0;
    }
    if (pos.y() < bound && v.get(pos).y() < 0) {
        v.get(pos).y() = 0;
    }
    if (pos.y() > (v.col - bound) && v.get(pos).y() > 0) {
        v.get(pos).y() = 0;
    }
}

void P2G(Scene &scene) {
    for (int i=0; i<scene.G.m.length; i++) {
        scene.G.m[i] = 0.0;
        scene.G.v[i] = Vec2::Zero();
    }
    for (int p=0; p<scene.P.x.length; p++) {
        Vec2 w[3];
        Vec2i origin;
        Vec2 x_to_origin;
        calculate_weight_parameters(scene.P.x, scene.G.dx, p, w, origin, x_to_origin);
        Mat2 affine = calculate_affine_matrix(scene, p);
        // double stress = -scene.dt * 4.0 * _E * _p_vol * (scene.P.J[p] - 1.0) / (scene.G.dx*scene.G.dx);
        // Mat2 diagonal = Eigen::DiagonalMatrix<double, 2>(stress, stress);
        // Mat2 affine = diagonal + scene.P.mass * scene.P.C[p];
        for (int i=0; i<3; i++) {
            for (int j=0; j<3; j++) {
                Vec2i offset = Vec2i(i, j);
                Vec2 dpos = (offset.cast<double>() - x_to_origin) * scene.G.dx;
                double weight = w[i].x() * w[j].y();
                scene.G.v.get(origin + offset) += weight * (scene.P.mass * scene.P.v[p] + affine * dpos);
                scene.G.m.get(origin + offset) += weight * scene.P.mass;
            }
        }
    }
    for (int i=0; i<scene.G.m.row; i++) {
        for (int j=0; j<scene.G.m.col; j++) {
            Vec2i pos = Vec2i(i, j);
            if (scene.G.m.get(pos) > 0.0) {
                scene.G.v.get(pos) /= scene.G.m.get(pos);
            }
            scene.G.v.get(pos).y() -= scene.dt * _gravity;
            boundary(scene.G.v, _bound, pos);
        }
    }
}

Mat2 outer_product(Vec2 &v, Vec2 &w) {
    Mat2 m;
    m << (v[0] * w[0]), (v[0] * w[1]),
         (v[1] * w[0]), (v[1] * w[1]);
    return m;
}

void G2P(Scene &scene) {
    for (int p=0; p<scene.P.x.length; p++) {
        Vec2 w[3];
        Vec2i origin;
        Vec2 x_to_origin;
        calculate_weight_parameters(scene.P.x, scene.G.dx, p, w, origin, x_to_origin);
        Vec2 new_v = Vec2::Zero();
        Mat2 new_C = Mat2::Zero();
        for (int i=0; i<3; i++) {
            for (int j=0; j<3; j++) {
                Vec2i offset = Vec2i(i, j);
                Vec2 dpos = (offset.cast<double>() - x_to_origin) * scene.G.dx;
                double weight = w[i].x() * w[j].y();
                Vec2 G_v = scene.G.v.get(origin+offset);
                new_v += weight * G_v;
                new_C += 4.0 * weight * scene.G.inv_dx_square * outer_product(G_v, dpos);
                // new_C += 4.0 * scene.G.inv_dx * weight * outer_product(G_v, dpos);
            }
        }
        scene.P.v[p] = new_v;
        scene.P.x[p] += scene.P.v[p] * scene.dt;
        scene.P.x[p] = clamp(scene.P.x[p], 0, 1);
        // scene.P.J[p] *= 1.0 + scene.dt * new_C.trace();
        scene.P.C[p] = new_C;
    }
}

EXPORT void step() {
    P2G(SCENE);
    G2P(SCENE);
}

EXPORT void test() {
    /* Eigen::Matrix3d A;
    A << 1, 0, 1,
         0, 1, 1,
         0, 0, 0;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(A, Eigen::ComputeThinU|Eigen::ComputeThinV);
    std::cout << svd.matrixU() << std::endl;
    std::cout << svd.singularValues() << std::endl;
    std::cout << svd.matrixV() << std::endl; */
    /* Mat2 m;
    m << 1, 2, 3, 4;
    Eigen::Matrix4d mm;
    std::cout << m << std::endl;
    int rows = 2;
    int cols = 2;
    mm.topLeftCorner(rows, cols) = m;
    mm.topRightCorner(rows, cols) = m;
    mm.bottomLeftCorner(rows, cols) = m;
    mm.bottomRightCorner(rows, cols) = m;
    std::cout << mm << std::endl; */
    Eigen::VectorXd v(8);
    v(0) = 1;
    v(7) = 8;
    std::cout << v << std::endl;
    std::cout << &v << std::endl;
    std::cout << &v(0) << std::endl;
    std::cout << &v(7) << std::endl;
    std::cout << v.size() << std::endl;
    double* v_data_ptr = &v(0);
    for (int i=0; i<8; i++) {
        std::cout << v_data_ptr[i] << std::endl;
    }
}
