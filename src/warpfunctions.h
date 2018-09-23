#pragma once

#include <glm/glm.hpp>
#include "utilities.h"

namespace Warp
{
  __host__ __device__ inline glm::vec3 SquareToDiskConcentric(float rngX, float rngY)
  {
    float radius, angle;

    float a = 2.0f * rngX - 1.0f;
    float b = 2.0f * rngY - 1.0f;

    if (a > -b)
    {
      if (a > b)
      {
        radius = a;
        angle = (Pi / 4.0f) * (b / a);
      }
      else
      {
        radius = b;
        angle = (Pi / 4.0f) * (2.0f - (a / b));
      }
    }
    else
    {
      if (a < b)
      {
        radius = -a;
        angle = (Pi / 4.0f) * (4.0f + (b / a));
      }
      else
      {
        radius = -b;
        if (b != 0)
        {
          angle = (Pi / 4.0f) * (6.0f - (a / b));
        }
        else
        {
          angle = 0.0f;
        }
      }
    }

    float finalX = radius * std::cos(angle);
    float finalY = radius * std::sin(angle);

    return glm::vec3(finalX, finalY, 0);
  }

  __host__ __device__ inline glm::vec3 SquareToHemisphereCosine(float rngX, float rngY)
  {
    glm::vec3 val = SquareToDiskConcentric(rngX, rngY);

    float finalZ = std::sqrt(std::fmax(0.0f, 1.0f - val.x * val.x - val.y * val.y));

    return glm::vec3(val.x, val.y, finalZ);
  }

  __host__ __device__ inline float SquareToHemisphereCosinePDF(const glm::vec3 &sample)
  {
    return InvPi * std::abs(sample.z);
  }
} // namespace Warp
