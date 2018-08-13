#include <iostream>
#include <limits>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>

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

class Triangle {
 public:
  Point3 points[3];
  Triangle(const Point3 p0, const Point3 p1, const Point3 p2)
    : points{p0, p1, p2} {};
};

class Polygon {
 public:
  std::vector<Triangle> triangles;
  explicit Polygon(const std::vector<Triangle> &triangles)
    : triangles(triangles) {}
  explicit Polygon(std::vector<Triangle> &&triangles)
    : triangles(std::move(triangles)) {}
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

Point3 operator*(const Mat4 &mat, const Point3 &p) {
  Point3 res;
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      res.v[i] += mat.v[i][j] * p.v[j];
    }
  }
  return res;
}

Triangle operator*(const Mat4 &mat, const Triangle &t) {
  return Triangle(mat * t.points[0], mat * t.points[1], mat * t.points[2]);
}

Polygon operator*(const Mat4 &mat, const Polygon &poly) {
  std::vector<Triangle> res;
  for (const auto &tri : poly.triangles) {
    res.push_back(mat * tri);
  }
  return Polygon(std::move(res));
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

Point3 perspective(const Mat4 &mat, const Point3 &p) {
  Point3 res3 = mat * p;
  return Point3(res3.v[0] / res3.v[3], res3.v[1] / res3.v[3], res3.v[2] / res3.v[3]);
}

Triangle perspective(const Mat4 &mat, const Triangle &tri) {
  return Triangle(
      perspective(mat, tri.points[0]),
      perspective(mat, tri.points[1]),
      perspective(mat, tri.points[2])
  );
}

Polygon perspective(const Mat4 &mat, const Polygon &poly) {
  std::vector<Triangle> res;
  for (const auto &tri : poly.triangles) {
    res.push_back(perspective(mat, tri));
  }
  return Polygon(std::move(res));
}

Polygon make_cube() {
  Point3 a0(0.0f, 0.0f, 0.0f);
  Point3 a1[3] = {
    Point3(0.0f, 0.0f, 1.0f), 
    Point3(0.0f, 1.0f, 0.0f), 
    Point3(1.0f, 0.0f, 0.0f)
  };
  Point3 a2[3] = {
    Point3(1.0f, 0.0f, 1.0f), 
    Point3(0.0f, 1.0f, 1.0f), 
    Point3(1.0f, 1.0f, 0.0f)
  };
  Point3 a3(1.0f, 1.0f, 1.0f);
  std::vector<Triangle> tri;
  for (size_t i = 0; i < 3; ++i) {
    tri.emplace_back(a0, a1[i], a1[(i+1)%3]);
    tri.emplace_back(a1[i], a1[(i+1)%3], a2[(i+1)%3]);
    tri.emplace_back(a1[i], a2[i], a2[(i+1)%3]);
    tri.emplace_back(a2[i], a2[(i+1)%3], a3);
  }
  return Polygon(tri);
}

Polygon poly_from_obj_file(const std::string &filename) {
  std::ifstream ifs(filename);
  std::vector<Point3> vp;
  std::vector<Triangle> tris;
  std::string line;
  while (std::getline(ifs, line)) {
    std::stringstream ss;
    ss << line;
    std::string type;
    ss >> type;
    if (type == "v") {
      float x, y, z;
      ss >> x >> y >> z;
      vp.emplace_back(x, y, z);
    } else if (type == "f") {
      int p1, p2, p3;
      ss >> p1 >> p2 >> p3;
      --p1; --p2; --p3;
      tris.emplace_back(vp[p1], vp[p2], vp[p3]);
    }
  }
  return Polygon(std::move(tris));
}

class Line {
 public:
  Point3 p[2];
  Line(Point3 p1, Point3 p2)
    : p{p1, p2} {}
};

std::tuple<bool, Point3> crossY(const Line l, const float y) {
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
    const float y, const float aspect, const Triangle &tri,
    const size_t index) {
  std::tuple<bool, Point3> crs[3];
  for (size_t i = 0; i < 3; ++i) {
    Point3 p1 = tri.points[i];
    Point3 p2 = tri.points[(i+1)%3];
    Line l(p1, p2);
    crs[i] = crossY(l, y);
  }
  float beta = aspect * width / 2;
  for (size_t i = 0; i < 3; ++i) {
    auto [valid1, crs1] = crs[i];
    auto [valid2, crs2] = crs[(i+1)%3];
    if (!valid1 || !valid2) continue;
    float x1 = crs1.v[0];
    float x2 = crs2.v[0];
    // z' = 1 + 1/z -> z = 1 / (z' - 1)
    float z1 = 1.0f / (crs1.v[2] - 1.0f);
    float z2 = 1.0f / (crs2.v[2] - 1.0f);
    if (z1 <= 0) continue;
    if (z2 <= 0) continue;
    if (x1 > x2) {
      std::swap(x1, x2);
      std::swap(z1, z2);
    }
    int begin = std::max(0.0f, ceilf(beta * (x1 + 1.0)));
    int end = std::min(width-1.0f, floorf(beta * (x2 + 1.0)));
    for (int j = begin; j <= end; ++j) {
      float x = (j / beta - 1.0);
      float z = z1 + (z2 - z1) / (x2 - x1) * (x - x1);
      if (z < depth[j]) {
        depth[j] = z;
        tri_index_line[j] = index;
      }
    }
  }
}

void rasterise(std::vector<int32_t> &tri_index_line, std::vector<float> &depth,
    const size_t width,
    const float y, const float aspect, const Polygon &poly) {
  for (size_t i = 0; i < poly.triangles.size(); ++i) {
    rasterise(tri_index_line, depth, width, y, aspect, poly.triangles[i], i);
  }
}

template <typename T>
using Buf = std::vector<std::vector<T>>;

void rasterise(Buf<int32_t> &tri_index, const size_t width, const size_t height,
    const Polygon &poly) {
#pragma omp parallel for
  for (size_t i = 0; i < height; ++i) {
    std::vector<float> depth(width, std::numeric_limits<float>::infinity());
    rasterise(tri_index[i], depth, width, 1.0 - 2 * i / (float)height,
        height / (float)width, poly);
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

Point3 normal_vector(const Triangle &tri) {
  return cross(Line(tri.points[1], tri.points[0]),
      Line(tri.points[2], tri.points[0]));
}

float dot(Point3 lhs, Point3 rhs) {
  return lhs.v[0] * rhs.v[0] + lhs.v[1] * rhs.v[1] + lhs.v[2] * rhs.v[2];
}

void draw(cv::Mat &baseimage, const Buf<int32_t> &tri_index,
    const size_t width, const size_t height, const Polygon &poly) {
  const Point3 light_dir = Point3::direction(1, 1, 1);
  for (size_t i = 0; i < height; ++i) {
    cv::Vec3b *line = baseimage.ptr<cv::Vec3b>(i);
    for (size_t j = 0; j < width; ++j) {
      int32_t index = tri_index[i][j];
      if (index < 0) continue;
      auto &tri = poly.triangles[index];
      Point3 normal = normal_vector(tri);
      float cos = dot(normal, light_dir)
        / sqrtf(dot(normal, normal) * dot(light_dir, light_dir));
      line[j][1] = 200 * std::max(0.0f, -cos) + 50;
    }
  }
}

int main() {
  constexpr size_t width = 640;
  constexpr size_t height = 480;
  int cnt = 0;
  Polygon cube = make_cube();
  Polygon teapot = poly_from_obj_file("teapot.obj");
  float rad = 0.0;
  cv::Mat baseimage(cv::Size(width, height), CV_8UC3);
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
      cnt = 0;
    }
    rad += 0.01;
    cv::waitKey(16);
  }
	return 0;
}
