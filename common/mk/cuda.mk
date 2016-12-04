# (c) 2007 The Board of Trustees of the University of Illinois.

# Default language wide options

# CUDA specific
LANG_CFLAGS=-I$(PARBOIL_ROOT)/common/include -I$(CUDA_PATH)/include
LANG_CXXFLAGS=$(LANG_CFLAGS)
LANG_LDFLAGS=-L$(CUDA_LIB_PATH)

LANG_CUDACFLAGS=$(LANG_CFLAGS)

CFLAGS=$(APP_CFLAGS) $(LANG_CFLAGS) $(PLATFORM_CFLAGS)
CXXFLAGS=$(APP_CXXFLAGS) $(LANG_CXXFLAGS) $(PLATFORM_CXXFLAGS)

CUDACFLAGS=$(LANG_CUDACFLAGS) $(PLATFORM_CUDACFLAGS) $(APP_CUDACFLAGS) 
CUDALDFLAGS=$(LANG_LDFLAGS) $(PLATFORM_CUDALDFLAGS) $(APP_CUDALDFLAGS)

# Rules common to all makefiles

########################################
# Functions
########################################

# Add BUILDDIR as a prefix to each element of $1
INBUILDDIR=$(addprefix $(BUILDDIR)/,$(1))

# Add SRCDIR as a prefix to each element of $1
INSRCDIR=$(addprefix $(SRCDIR)/,$(1))


########################################
# Environment variable check
########################################

# The second-last directory in the $(BUILDDIR) path
# must have the name "build".  This reduces the risk of terrible
# accidents if paths are not set up correctly.
ifeq ("$(notdir $(BUILDDIR))", "")
$(error $$BUILDDIR is not set correctly)
endif

ifneq ("$(notdir $(patsubst %/,%,$(dir $(BUILDDIR))))", "build")
$(error $$BUILDDIR is not set correctly)
endif

.PHONY: run

ifeq ($(CUDA_PATH),)
FAILSAFE=no_cuda
else 
FAILSAFE=
endif

########################################
# Derived variables
########################################

ifeq ($(DEBUGGER),)
DEBUGGER=gdb
endif

OBJS = $(call INBUILDDIR,$(SRCDIR_OBJS))

########################################
# Rules
########################################

default: $(FAILSAFE) $(BUILDDIR) $(BIN)

run:
	@echo "Resolving CUDA runtime library..."
	@$(shell echo $(RUNTIME_ENV)) nvprof --log-file Gpu_metric -m l1_cache_global_hit_rate,l1_cache_local_hit_rate,sm_efficiency,ipc,achieved_occupancy,sm_efficiency_instance,ipc_instance,dram_read_throughput,dram_write_throughput,shared_efficiency,gld_efficiency,gst_efficiency,warp_execution_efficiency,issued_ipc,issue_slot_utilization,local_load_transactions,local_store_transactions,shared_load_transactions,shared_store_transactions,gld_transactions,gst_transactions,sysmem_read_transactions,sysmem_write_transactions,dram_read_transactions,dram_write_transactions,l2_read_transactions,l2_write_transactions,warp_nonpred_execution_efficiency,cf_executed,ldst_executed,flop_count_sp,flop_count_sp_add,flop_count_sp_mul,flop_count_sp_fma,flop_count_dp,flop_count_dp_add,flop_count_dp_mul,flop_count_dp_fma,flop_count_sp_special,stall_inst_fetch,stall_exec_dependency,stall_memory_dependency,stall_sync,stall_other,l1_shared_utilization,l2_utilization,tex_utilization,dram_utilization,sysmem_utilization,ldst_fu_utilization,alu_fu_utilization,cf_fu_utilization,tex_fu_utilization,inst_executed,issue_slots,l2_atomic_throughput,inst_fp_32,inst_fp_64,inst_integer,inst_bit_convert,inst_control,inst_compute_ld_st,inst_misc,inst_inter_thread_communication,atomic_transactions,atomic_transactions_per_request,l2_l1_read_transactions,l2_l1_write_transactions,l2_atomic_transactions,stall_pipe_busy,flop_sp_efficiency,flop_dp_efficiency,stall_memory_throttle,eligible_warps_per_cycle,atomic_throughput $(BIN) $(ARGS)

debug:
	@echo "Resolving CUDA runtime library..."
	@$(shell echo $(RUNTIME_ENV)) LD_LIBRARY_PATH=$(CUDA_LIB_PATH) ldd $(BIN) | grep cuda
	@$(shell echo $(RUNTIME_ENV)) LD_LIBRARY_PATH=$(CUDA_LIB_PATH) $(DEBUGGER) --args $(BIN) $(ARGS)

clean :
	rm -rf $(BUILDDIR)/*
	if [ -d $(BUILDDIR) ]; then rmdir $(BUILDDIR); fi

$(BIN) : $(OBJS) $(BUILDDIR)/parboil_cuda.o
	$(CUDALINK) $^ -o $@ $(CUDALDFLAGS)

$(BUILDDIR) :
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/%.o : $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/parboil_cuda.o: $(PARBOIL_ROOT)/common/src/parboil_cuda.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.cu
	$(CUDACC) $< $(CUDACFLAGS) -c -o $@

no_cuda:
	@echo "CUDA_PATH is not set. Open $(CUDA_ROOT)/common/Makefile.conf to set default value."
	@echo "You may use $(PLATFORM_MK) if you want a platform specific configurations."
	@exit 1

