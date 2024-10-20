#pragma once
#include "eigen_type.h"

template <typename T>
class Field {
public:
    T* data;
    int row;
    int col;
    int length;
    Field(): data(0), row(0), col(0), length(0) {}
    Field(T* data, int length): data(data), row(length), col(1), length(length) {}
    Field(T* data, int row, int col): data(data), row(row), col(col), length(row*col) {}
    T& operator[](int i) const {
        return data[i];
    };
    T& get(Vec2i i) const {
        int index = i.x() * col + i.y();
        return data[index];
    }
    T& get(int i, int j) const {
        return data[i * col + j];
    }
};
