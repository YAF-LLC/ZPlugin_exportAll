//
// Copyright (C) 2023 Kazutaka Nakashima (kazutaka.nakashima@n-taka.info)
//
// GPLv3
//
// This file is part of exportAll.
//
// exportAll is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
//
// exportAll is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with exportAll. If not, see <https://www.gnu.org/licenses/>.
//

#include "convert.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <filesystem>
namespace fs = std::filesystem;

#include "readGoZAndTriangulate.h"
#include "igl/write_triangle_mesh.h"

extern "C" DLLEXPORT float convert(char *someText, double optValue, char *outputBuffer, int optBuffer1Size, char *pOptBuffer2, int optBuffer2Size, char **zData)
{
    ////
    // parse parameter (JSON)
    fs::path jsonPath(someText);
    std::ifstream ifs(jsonPath);
    nlohmann::json json = nlohmann::json::parse(ifs);
    ifs.close();

    const std::string rootString = json.at("root").get<std::string>();
    const fs::path rootPath(rootString);
    fs::path dataDirectory(rootPath);
    dataDirectory /= "data";
    dataDirectory.make_preferred();

    const std::string targetDirectoryString = json.at("targetDirectory").get<std::string>();
    fs::path targetDirectory(targetDirectoryString);
    targetDirectory.make_preferred();

    const std::string formatToBeExported = json.at("format").get<std::string>();
    const bool separate = json.at("separate").get<bool>();
    const bool bin = json.at("bin").get<bool>();

    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> VVec;
    std::vector<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>> FVec;
    std::vector<fs::path> destPathVec;

    // load add GoZ meshes to memory
    for (const fs::directory_entry &entry : fs::directory_iterator(dataDirectory))
    {
        const fs::path ext = entry.path().extension();
        std::string extString = ext.string();
        std::transform(extString.begin(), extString.end(), extString.begin(), ::tolower);

        std::string destFileName = entry.path().stem().string();
        destFileName += formatToBeExported;
        fs::path destPath(targetDirectory);
        destPath /= destFileName;

        if (extString == ".goz")
        {
            std::string meshName;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> V;
            Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> F;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> UV_u;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> UV_v;
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> VC;
            Eigen::Matrix<double, Eigen::Dynamic, 1> M;
            Eigen::Matrix<int, Eigen::Dynamic, 1> G;
            readGoZAndTriangulate(entry.path(), meshName, V, F, UV_u, UV_v, VC, M, G);

            VVec.push_back(V);
            FVec.push_back(F);
            destPathVec.push_back(destPath);
        }
    }

    if (!separate)
    {
        // merge all meshes
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mergedV;
        mergedV.resize(0, 3);
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> mergedF;
        mergedF.resize(0, 3);
        for (int meshIdx = 0; meshIdx < VVec.size(); ++meshIdx)
        {
            const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &V = VVec.at(meshIdx);
            // here, we "break" original F, but this is acceptable because we erase original F later
            Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> &F = FVec.at(meshIdx);

            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> tmpV = std::move(mergedV);
            Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> tmpF = std::move(mergedF);
            mergedV.resize(tmpV.rows() + V.rows(), tmpV.cols());
            mergedF.resize(tmpF.rows() + F.rows(), tmpF.cols());

            mergedV << tmpV,
                V;
            mergedF << tmpF,
                F.array() + tmpV.rows();
        }
        VVec.clear();
        VVec.push_back(mergedV);
        FVec.clear();
        FVec.push_back(mergedF);

        // use active subtool name for export
        std::string destFileName = json.at("activeSubTName").get<std::string>();
        destFileName += formatToBeExported;
        fs::path destPath(targetDirectory);
        destPath /= destFileName;
        destPathVec.clear();
        destPathVec.push_back(destPath);
    }

    for (int meshIdx = 0; meshIdx < VVec.size(); ++meshIdx)
    {
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &V = VVec.at(meshIdx);
        const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> &F = FVec.at(meshIdx);
        const fs::path &destPath = destPathVec.at(meshIdx);

        igl::write_triangle_mesh(destPath.string(), V, F, (bin ? igl::FileEncoding::Binary : igl::FileEncoding::Ascii));
    }

    // emptify
    fs::remove_all(dataDirectory);
    fs::create_directory(dataDirectory);

    return 1.0f;
}
