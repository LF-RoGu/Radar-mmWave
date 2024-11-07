#ifndef IWR6843_H
#define IWR6843_H

#pragma once
#include <stdio.h>
#include <string.h>
#include <fcntl.h> // File control definitions
#include <termios.h> // POSIX terminal control definitions
#include <unistd.h> // UNIX standard function definitions
#include <errno.h> // Error number definitions
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <stdint.h>

using namespace std;

namespace Modules
{
	class IWR6843
	{
		private:

		public:
			IWR6843();
	};
}

#endif // !IWR6843_H