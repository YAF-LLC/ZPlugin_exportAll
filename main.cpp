//
// Copyright (C) 2023 Kazutaka Nakashima (kazutaka.nakashima@n-taka.info)
// 
// GPLv3
//
// This file is part of offsetMaster.
// 
// offsetMaster is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
// 
// offsetMaster is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License along with offsetMaster. If not, see <https://www.gnu.org/licenses/>.
//

#include <fstream>
#include <string>
#include "nlohmann/json.hpp"

#include "convert.h"

int main()
{
    nlohmann::json json;
    json["root"] = ".";
    json["targetDirectory"] = ".";
    json["format"] = "ply";
    json["separate"] = true;
    json["activeSubTName"] = "Dog";
    json["bin"] = true;

    std::ofstream ofs("build/Release/parameters.txt");
    ofs << json.dump(4) << std::endl;
    ofs.close();

    char buf[255];
    char **zData = nullptr;
    convert("ZPlugs64/exportAll_2022/data/parameters.txt", 0.0, buf, 0, buf, 0, zData);
    return 0;

    // char buf[255];
    // char **zData = nullptr;
    // resample("/APPLICATIONS/ZBRUSH 2022/ZSTARTUP/ZPlugs64/remeshWithDualContouring_2022/data/parameters.txt", 0.0, buf, 0, buf, 0, zData);
    // return 0;
}
