#ifndef CONFIG_H_INCLUDE
#define CONFIG_H_INCLUDE


#include <cstddef>

enum Reg_type {L1=1,L2};

class Config
{
	public:
		static Config* GetInstance()
		{
			if (instance==NULL)
			{
				instance = new Config();
			}	
			return instance;
		}

		int iteration;
		Reg_type type;

	private:
		static Config* instance;
		Config()
		{
			iteration = 10;
			type = Reg_type::L1;
		}
};

#endif
