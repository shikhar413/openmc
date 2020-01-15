#!/bin/bash

dir=1d-homog-analog
mkdir -p $dir
cp base/1d-materials.xml $dir/materials.xml
cp base/1d-geometry.xml $dir/geometry.xml
cp base/1d-tallies.xml $dir/tallies.xml
cp base/1d-settings-se.xml $dir/settings.xml
cp base/*.py $dir/

dir=1d-homog-collision
mkdir -p $dir
cp base/1d-materials.xml $dir/materials.xml
cp base/1d-geometry.xml $dir/geometry.xml
cp base/1d-tallies.xml $dir/tallies.xml
cp base/1d-settings-se.xml $dir/settings.xml
cp base/*.py $dir/

dir=1d-homog-fb
mkdir -p $dir
cp base/1d-materials.xml $dir/materials.xml
cp base/1d-geometry.xml $dir/geometry.xml
cp base/1d-settings-fb.xml $dir/settings.xml
cp base/*.py $dir/

dir=1d-homog-nofet
mkdir -p $dir
cp base/1d-materials.xml $dir/materials.xml
cp base/1d-geometry.xml $dir/geometry.xml
cp base/1d-settings-fb.xml $dir/settings.xml
cp base/*.py $dir/

dir=1d-homog-nose
mkdir -p $dir
cp base/1d-materials.xml $dir/materials.xml
cp base/1d-geometry.xml $dir/geometry.xml
cp base/1d-settings-nose.xml $dir/settings.xml
cp base/*.py $dir/

dir=2d-beavrs-analog
mkdir -p $dir
cp base/2d-materials.xml $dir/materials.xml
cp base/2d-geometry.xml $dir/geometry.xml
cp base/2d-tallies.xml $dir/tallies.xml
cp base/2d-settings-se.xml $dir/settings.xml
cp base/*.py $dir/

dir=2d-beavrs-collision
mkdir -p $dir
cp base/2d-materials.xml $dir/materials.xml
cp base/2d-geometry.xml $dir/geometry.xml
cp base/2d-tallies.xml $dir/tallies.xml
cp base/2d-settings-se.xml $dir/settings.xml
cp base/*.py $dir/

dir=2d-beavrs-fb
mkdir -p $dir
cp base/2d-materials.xml $dir/materials.xml
cp base/2d-geometry.xml $dir/geometry.xml
cp base/2d-settings-fb.xml $dir/settings.xml
cp base/*.py $dir/

dir=2d-beavrs-nofet
mkdir -p $dir
cp base/2d-materials.xml $dir/materials.xml
cp base/2d-geometry.xml $dir/geometry.xml
cp base/2d-settings-fb.xml $dir/settings.xml
cp base/*.py $dir/

dir=2d-beavrs-nose
mkdir -p $dir
cp base/2d-materials.xml $dir/materials.xml
cp base/2d-geometry.xml $dir/geometry.xml
cp base/2d-settings-nose.xml $dir/settings.xml
cp base/*.py $dir/
