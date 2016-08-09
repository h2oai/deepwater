#!/bin/bash
if [ ! -f mnist.zip ]; then
    echo "Download mnist dataset";
    wget http://webdocs.cs.ualberta.ca/\~bx3/data/mnist.zip
    unzip mnist.zip
fi
