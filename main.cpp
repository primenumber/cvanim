#include <cmath>
#include <array>
#include <chrono>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <omp.h>

class Point3 {
 public:
  float v[4];
  Point3()
    : v{0, 0, 0, 1} {}
  Point3(const float x, const float y, const float z)
    : v{x, y, z, 1} {}
  Point3(const float x, const float y, const float z, const float w)
    : v{x, y, z, w} {}
  static Point3 direction(const float x, const float y, const float z) {
    return Point3(x, y, z, 0);
  }
};

Point3 operator*(const float lhs, const Point3& rhs) {
  return Point3(lhs * rhs.v[0], lhs * rhs.v[1], lhs * rhs.v[2]);
}

using idx_t = uint32_t;

class Triangle {
 public:
  idx_t indices[3];
  Triangle(const idx_t i0, const idx_t i1, const idx_t i2)
    : indices{i0, i1, i2} {};
};

class Polygon {
 public:
  std::vector<float> v[4];
  std::vector<Triangle> triangles;
  Polygon() = default;
  Polygon(const Polygon&) = default;
  Polygon(Polygon&&) = default;
  Polygon(
      const std::vector<float>& vx,
      const std::vector<float>& vy,
      const std::vector<float>& vz,
      const std::vector<float>& vw,
      const std::vector<Triangle>& triangles)
    : v({vx, vy, vz, vw}), triangles(triangles) {}
  size_t num_vertices() const { return std::size(v[0]); }
  size_t num_triangles() const { return std::size(triangles); }
};

class Mat4 {
 public:
  float v[4][4];
  static Mat4 zero() {
    Mat4 res;
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        res.v[i][j] = 0.0;
      }
    }
    return res;
  }
  static Mat4 id() {
    Mat4 res;
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        if (i != j) {
          res.v[i][j] = 0.0;
        } else {
          res.v[i][j] = 1.0;
        }
      }
    }
    return res;
  }
};

Polygon operator*(const Mat4 &mat, const Polygon &poly) {
  size_t n = poly.num_vertices();
  Polygon res = poly;
  for (size_t i = 0; i < 4; ++i) {
    res.v[i].assign(n, 0.f);
  }
  for (size_t j = 0; j < 4; ++j) {
    for (size_t i = 0; i < n; ++i) {
      for (size_t k = 0; k < 4; ++k) {
        res.v[j][i] += mat.v[j][k] * poly.v[k][i];
      }
    }
  }
  return res;
}

Mat4 operator*(const Mat4 &lhs, const Mat4 &rhs) {
  Mat4 res = Mat4::zero();
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      for (size_t k = 0; k < 4; ++k) {
        res.v[i][j] += lhs.v[i][k] * rhs.v[k][j];
      }
    }
  }
  return res;
}

Mat4 translate(const float tx, const float ty, const float tz) {
  Mat4 res = Mat4::id();
  res.v[0][3] = tx;
  res.v[1][3] = ty;
  res.v[2][3] = tz;
  return res;
}

Mat4 scale(const float sx, const float sy, const float sz) {
  Mat4 res = Mat4::id();
  res.v[0][0] = sx;
  res.v[1][1] = sy;
  res.v[2][2] = sz;
  return res;
}

Mat4 rotateX(const float rad) {
  Mat4 res = Mat4::id();
  res.v[1][1] = cos(rad);
  res.v[1][2] = sin(rad);
  res.v[2][1] = -sin(rad);
  res.v[2][2] = cos(rad);
  return res;
}

Mat4 rotateY(const float rad) {
  Mat4 res = Mat4::id();
  res.v[2][2] = cos(rad);
  res.v[2][0] = sin(rad);
  res.v[0][2] = -sin(rad);
  res.v[0][0] = cos(rad);
  return res;
}

Mat4 rotateZ(const float rad) {
  Mat4 res = Mat4::id();
  res.v[0][0] = cos(rad);
  res.v[0][1] = sin(rad);
  res.v[1][0] = -sin(rad);
  res.v[1][1] = cos(rad);
  return res;
}

Mat4 perspective() {
  Mat4 res = Mat4::id();
  res.v[2][3] = 1.0f;
  res.v[3][2] = 1.0f;
  res.v[3][3] = 0.0f;
  return res;
}

Polygon perspective(const Mat4 &mat, const Polygon &poly) {
  auto persed = mat * poly;
  size_t n = poly.num_vertices();
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      persed.v[j][i] /= persed.v[3][i];
    }
  }
  return persed;
}

//Polygon make_cube() {
//  Point3 a0(0.0f, 0.0f, 0.0f);
//  Point3 a1[3] = {
//    Point3(0.0f, 0.0f, 1.0f), 
//    Point3(0.0f, 1.0f, 0.0f), 
//    Point3(1.0f, 0.0f, 0.0f)
//  };
//  Point3 a2[3] = {
//    Point3(1.0f, 0.0f, 1.0f), 
//    Point3(0.0f, 1.0f, 1.0f), 
//    Point3(1.0f, 1.0f, 0.0f)
//  };
//  Point3 a3(1.0f, 1.0f, 1.0f);
//  std::vector<Triangle> tri;
//  for (size_t i = 0; i < 3; ++i) {
//    tri.emplace_back(a0, a1[i], a1[(i+1)%3]);
//    tri.emplace_back(a1[i], a1[(i+1)%3], a2[(i+1)%3]);
//    tri.emplace_back(a1[i], a2[i], a2[(i+1)%3]);
//    tri.emplace_back(a2[i], a2[(i+1)%3], a3);
//  }
//  return Polygon(tri);
//}

Polygon poly_from_obj_file(const std::string &filename) {
  std::ifstream ifs(filename);
  Polygon poly;
  std::string line;
  while (std::getline(ifs, line)) {
    std::stringstream ss;
    ss << line;
    std::string type;
    ss >> type;
    if (type == "v") {
      float x, y, z;
      ss >> x >> y >> z;
      poly.v[0].push_back(x);
      poly.v[1].push_back(y);
      poly.v[2].push_back(z);
      poly.v[3].push_back(1.f);
    } else if (type == "f") {
      int p1, p2, p3;
      ss >> p1 >> p2 >> p3;
      --p1; --p2; --p3;
      poly.triangles.emplace_back(p1, p2, p3);
    }
  }
  return poly;
}

class Line {
 public:
  Point3 p[2];
  Line(Point3 p1, Point3 p2)
    : p{p1, p2} {}
};

std::tuple<bool, Point3> crossY(const Line& l, const float y) {
  float x1 = l.p[0].v[0];
  float x2 = l.p[1].v[0];
  float y1 = l.p[0].v[1];
  float y2 = l.p[1].v[1];
  float z1 = l.p[0].v[2];
  float z2 = l.p[1].v[2];
  float alpha = (y - y1) / (y2 - y1);
  return std::make_tuple(
      0 <= alpha && alpha <= 1,
      Point3(alpha * (x2 - x1) + x1, y, alpha * (z2 - z1) + z1)
  );
}

void rasterise(std::vector<int32_t> &tri_index_line, std::vector<float> &depth,
    const size_t width,
    const float y, const float aspect,
    const std::array<Point3, 3> &tri,
    const size_t index) {
  std::tuple<bool, Point3> crs[3];
  for (size_t i = 0; i < 3; ++i) {
    Point3 p1 = tri[i];
    const auto nx = i < 2 ? i+1 : 0;
    Point3 p2 = tri[nx];
    Line l(p1, p2);
    crs[i] = crossY(l, y);
  }
  float beta = aspect * width / 2.0f;
  for (size_t i = 0; i < 3; ++i) {
    auto [valid1, crs1] = crs[i];
    const auto nx = i < 2 ? i+1 : 0;
    auto [valid2, crs2] = crs[nx];
    if (!valid1 || !valid2) continue;
    float x1 = beta * crs1.v[0];
    float x2 = beta * crs2.v[0];
    // z' = 1 + 1/z -> z = 1 / (z' - 1)
    float z1 = beta / (crs1.v[2] - 1.0f);
    if (z1 <= 0) continue;
    float z2 = beta / (crs2.v[2] - 1.0f);
    if (z2 <= 0) continue;
    if (x1 > x2) {
      std::swap(x1, x2);
      std::swap(z1, z2);
    }
    int begin = std::max(0.0f, ceilf(x1 + beta));
    int end = std::min(width-1.0f, floorf(x2 + beta));
    auto coeff = (z2 - z1) / (x2 - x1);
    auto bias = z1 - coeff * (beta + x1);
    for (int j = begin; j <= end; ++j) {
      float x = static_cast<float>(j);
      float z = coeff * x + bias;
      if (z < depth[j]) {
        depth[j] = z;
        tri_index_line[j] = index;
      }
    }
  }
}

template <typename T>
using Buf = std::vector<std::vector<T>>;

void calc_ranges(std::vector<std::tuple<float, float, size_t>> &ranges, const Polygon &poly) {
#pragma omp parallel for
  for (size_t i = 0; i < poly.num_triangles(); ++i) {
    float min_y = std::numeric_limits<float>::infinity();
    float max_y = -std::numeric_limits<float>::infinity();
    for (size_t j = 0; j < 3; ++j) {
      auto idx = poly.triangles[i].indices[j];
      min_y = std::min(min_y, poly.v[1][idx]);
      max_y = std::max(max_y, poly.v[1][idx]);
    }
    ranges[i] = std::make_tuple(min_y, max_y, i);
  }
  std::sort(std::begin(ranges), std::end(ranges));
}

void rasterise(Buf<int32_t> &tri_index, const size_t width, const size_t height,
    const Polygon &poly) {
  std::vector<std::tuple<float, float, size_t>> ranges(poly.num_triangles());
  calc_ranges(ranges, poly);

#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < height; ++i) {
    std::vector depth(width, std::numeric_limits<float>::infinity());
    const float y = 1.0f - static_cast<float>(2 * i) / static_cast<float>(height);
    const float aspect = static_cast<float>(height) / static_cast<float>(width);
    for (auto&& [y_start, y_end, j] : ranges) {
      if (y_start > y) break;
      if (y_end < y) continue;
      std::array<Point3, 3> tri;
      for (size_t k = 0; k < 3; ++k) {
        auto idx = poly.triangles[j].indices[k];
        for (size_t l = 0; l < 3; ++l) {
          tri[k].v[l] = poly.v[l][idx];
        }
      }
      rasterise(tri_index[i], depth, width, y, aspect, tri, j);
    }
  }
}

Point3 cross(const Line &lhs, const Line &rhs) {
  float x1 = lhs.p[1].v[0] - lhs.p[0].v[0];
  float y1 = lhs.p[1].v[1] - lhs.p[0].v[1];
  float z1 = lhs.p[1].v[2] - lhs.p[0].v[2];
  float x2 = rhs.p[1].v[0] - rhs.p[0].v[0];
  float y2 = rhs.p[1].v[1] - rhs.p[0].v[1];
  float z2 = rhs.p[1].v[2] - rhs.p[0].v[2];
  return Point3(y1*z2 - y2*z1, z1*x2 - z2*x1, x1*y2 - x2*y1);
}

Point3 normal_vector(const std::array<Point3, 3> &tri) {
  return cross(Line(tri[1], tri[0]),
      Line(tri[2], tri[0]));
}

float dot(const Point3& lhs, const Point3& rhs) {
  return lhs.v[0] * rhs.v[0] + lhs.v[1] * rhs.v[1] + lhs.v[2] * rhs.v[2];
}

void draw(cv::Mat &baseimage, const Buf<int32_t> &tri_index,
    const size_t width, const size_t height, const Polygon &poly) {
  const Point3 light_dir = (1.f / std::sqrt(3.f)) * Point3::direction(1, 1, 1);
#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < height; ++i) {
    cv::Vec3b *line = baseimage.ptr<cv::Vec3b>(i);
    for (size_t j = 0; j < width; ++j) {
      int32_t index = tri_index[i][j];
      if (index < 0) continue;
      std::array<Point3, 3> tri;
      for (size_t k = 0; k < 3; ++k) {
        auto point_index = poly.triangles[index].indices[k];
        for (size_t l = 0; l < 3; ++l) {
          tri[k].v[l] = poly.v[l][point_index];
        }
      }
      Point3 normal = normal_vector(tri);
      float cos = dot(normal, light_dir)
        / sqrtf(dot(normal, normal));
      line[j][1] = 200.0f * std::max(0.0f, -cos) + 50.0f;
    }
  }
}

int main() {
  constexpr size_t width = 1920;
  constexpr size_t height = 1080;
  const size_t fps = 144;
  int cnt = 0;
  //Polygon cube = make_cube();
  Polygon teapot = poly_from_obj_file("teapot.obj");
  float rad = 0.0;
  cv::Mat baseimage(cv::Size(width, height), CV_8UC3);
  const auto start = std::chrono::system_clock::now();
  while (true) {
    std::cerr << cnt << std::endl;
    Mat4 mat_scale = scale(30.0f, 30.0f, 30.0f);
    Mat4 mat_rot = rotateY(rad);
    Mat4 mat_trans = translate(-50.0f, -50.0f, 200.0f);
    Polygon teapot_transed = (mat_trans * mat_rot * mat_scale) * teapot;
    Polygon teapot_persed = perspective(perspective(), teapot_transed);
#pragma omp parallel for
    for (size_t i = 0; i < height; ++i) {
      cv::Vec3b *ptr = baseimage.ptr<cv::Vec3b>(i);
      for (size_t j = 0; j < width; ++j) {
        ptr[j][0] = 0;
        ptr[j][1] = 0;
        ptr[j][2] = 0;
      }
    }
    Buf<int32_t> tri_index(height, std::vector<int32_t>(width, -1));
    rasterise(tri_index, width, height, teapot_persed);
    draw(baseimage, tri_index, width, height, teapot_transed);
    cv::imshow("Image", baseimage);
    ++cnt;
    if (cnt >= 256) {
      break;
      cnt = 0;
    }
    rad += 0.01;
    const auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    int rem = cnt * 1000 / fps - duration.count();
    cv::waitKey(std::max(1, rem));
  }
	return 0;
}
