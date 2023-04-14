#pragma once
#include "hdl_graph_slam/view/interactive_graph_view.hpp"
#include "hdl_graph_slam/information_matrix_calculator.hpp"
#include <pcl/search/kdtree.h>
#include <pcl/common/transforms.h>

namespace hdl_graph_slam {

class UpdateGraphFunction {
public:
  UpdateGraphFunction(std::shared_ptr<InteractiveGraphView>& graph): 
  graph_(graph), 
  run_{false},
  graph_loaded_{false},
  view_fp_keyframes_{false},
  view_tp_keyframes_{false},
  fp_percent_threshold_{0},
  tp_percent_threshold_{0},
  dist_threshold_{0.0f},
  progress_{"update graph function"} {
    update_dump_.reset(new InteractiveGraphView());
    update_dump_->init_gl();
  }
  ~UpdateGraphFunction() {}

  void run() { run_ = true; load_graph_ = true; }
  void close() { 
    run_ = false;
    update_dump_.reset(new InteractiveGraphView());
    update_dump_->init_gl();
  }

  void draw_ui() {
    if (!run_) {
      return;
    }

    load_dump_(load_graph_);
    load_graph_ = false;

    ImGui::Begin("window", &run_, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::DragFloat("Nearest distance threshold", &dist_threshold_, 0.1f, 0.1f, 10.0f);
    ImGui::DragInt("FP percent", &fp_percent_threshold_, 1, 0, 100);
    ImGui::DragInt("TP percent", &tp_percent_threshold_, 1, 0, 100);
    ImGui::Checkbox("view fp keyframes", &view_fp_keyframes_);
    ImGui::Checkbox("view tp keyframes", &view_tp_keyframes_);

    if (ImGui::Button("Caclulate FP")) {
      progress_.open<bool>("Caclulate FP", [=](guik::ProgressInterface& p) {
        caclulate_scores_(p);
        return true;
      });
    }

    if (ImGui::Button("Caclulate TP")) {
      progress_.open<bool>("Caclulate TP", [=](guik::ProgressInterface& p) {
        calculate_tp_scores_(p);
        return true;
      });
    }

    if (ImGui::Button("Delete TP")) {
      progress_.open<bool>("Delete TP", [=](guik::ProgressInterface& p) {
        detele_tp_(p);
        return true;
      });
    }

    if(progress_.run("Caclulate FP")) {
      std::cout << "done" << std::endl;
    }
    if(progress_.run("Caclulate TP")) {
      std::cout << "done" << std::endl;
    }
    if(progress_.run("Delete TP")) {
      std::cout << "done" << std::endl;
    }

    ImGui::End();
  }

  void draw_gl(glk::GLSLShader& shader, DrawFlags &draw_flags) {
    if (!run_) {
      return;
    }

    update_dump_->draw(draw_flags, shader);

    if (view_fp_keyframes_) {
      for (const auto& [id, score] : keyframes_fp_score_) {
        if (score >= fp_percent_threshold_) {
          const auto& keyframe = update_dump_->keyframes.at(id);
          auto& keyframe_view = update_dump_->keyframes_view_map[keyframe];
    
          shader.set_uniform("point_scale", 2.0f);
          keyframe_view->draw(draw_flags, shader, Eigen::Vector4f(0.0f, 0.0f, 1.0f, 1.0f), keyframe_view->lock()->estimate().matrix().cast<float>());
        }
      }
    }

    if (view_tp_keyframes_) {
      for (const auto& [id, tp] : keyframes_tp_score_) {
        if (tp <= tp_percent_threshold_) {
          const auto& keyframe = graph_->keyframes.at(id);
          auto& keyframe_view = graph_->keyframes_view_map[keyframe];
    
          shader.set_uniform("point_scale", 2.0f);
          keyframe_view->draw(draw_flags, shader, Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f), keyframe_view->lock()->estimate().matrix().cast<float>());
        }
      }

    }
  }

private:
  std::shared_ptr<InteractiveGraphView>& graph_;
  std::shared_ptr<InteractiveGraphView> update_dump_;
  bool run_;
  bool graph_loaded_;
  bool load_graph_;
  guik::ProgressModal progress_;

  pcl::PointCloud<pcl::PointXYZI>::Ptr map_;

  pcl::search::KdTree<pcl::PointXYZI>::Ptr tree_;

  float dist_threshold_;
  int fp_percent_threshold_;
  int tp_percent_threshold_;

  bool view_fp_keyframes_;
  bool view_tp_keyframes_;

  std::unordered_map<int, int> keyframes_fp_score_;
  std::unordered_map<int, int> keyframes_tp_score_;

private:

  // void add_keyframe_to_graph_() {
  //   graph_->add_ve
  // }

  void generate_map_() {
    map_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    map_->reserve(graph_->keyframes.begin()->second->cloud->size() * graph_->keyframes.size());

    for (const auto& keyframe : graph_->keyframes) {
      pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>());
      pcl::transformPointCloud(*(keyframe.second->cloud), *transformed_cloud, keyframe.second->node->estimate().matrix().cast<float>());
      std::copy(transformed_cloud->begin(), transformed_cloud->end(), std::back_inserter(map_->points));
    }
  }

  void detele_tp_(guik::ProgressInterface& progress) {
    progress.set_maximum(keyframes_tp_score_.size());
    for (const auto& [id, tp] : keyframes_tp_score_) {
      if (tp <= tp_percent_threshold_) {
        const auto& keyframe = graph_->keyframes.at(id);
        auto& keyframe_view = graph_->keyframes_view_map[keyframe];
        if (graph_->vertices_view_map.find(id) != graph_->vertices_view_map.end()) {
          graph_->delete_vertex(graph_->vertices_view_map.at(id));
        }
      }
      
      progress.increment();
    }
  }

  void calculate_tp_scores_(guik::ProgressInterface& progress) {
    progress.set_maximum(update_dump_->keyframes.size());
    for (const auto& [id, fp] : keyframes_fp_score_) {
      if (fp >= fp_percent_threshold_) {
        const auto& keyframe_fp = update_dump_->keyframes.at(id);
        
        pcl::PointCloud<pcl::PointXYZI>::Ptr keyframe_fp_cloud(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::transformPointCloud(*(keyframe_fp->cloud), *keyframe_fp_cloud, keyframe_fp->node->estimate().matrix().cast<float>());
        
        for (const auto& [g_id, keyframe] : graph_->keyframes) {
          pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>());
          pcl::transformPointCloud(*(keyframe->cloud), *transformed_cloud, keyframe->node->estimate().matrix().cast<float>());
          
          int tp = calculate_tp_(keyframe_fp_cloud, transformed_cloud);
          
          if (keyframes_tp_score_.find(id) == keyframes_tp_score_.end()) {
            keyframes_tp_score_.insert(std::make_pair(g_id, tp));
          }
        }
      }
      progress.increment();
    }
  }

  void caclulate_scores_(guik::ProgressInterface& progress) {
    generate_map_();

    keyframes_fp_score_.clear();

    tree_.reset(new pcl::search::KdTree<pcl::PointXYZI>());
    tree_->setInputCloud(map_);
    progress.set_maximum(update_dump_->keyframes.size());
    for (const auto& [id, keyframe] : update_dump_->keyframes) {
      pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>());
      pcl::transformPointCloud(*(keyframe->cloud), *transformed_cloud, keyframe->node->estimate().matrix().cast<float>());
      int score = calculate_fp_(transformed_cloud);
      progress.increment();
      if (keyframes_fp_score_.find(id) == keyframes_fp_score_.end()) {
        keyframes_fp_score_.insert(std::make_pair(id, score));
      }
    }

  }

  int calculate_fp_(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud) {
    std::vector<int> nn_indices(1);
    std::vector<float> nn_dists(1);

    // For each point in the source dataset
    int nr = 0;
    for(size_t i = 0; i < cloud->points.size(); ++i) {
      // Find its nearest neighbor in the target
      tree_->nearestKSearch(cloud->points[i], 1, nn_indices, nn_dists);

      // Deal with occlusions (incomplete targets)
      if(nn_dists[0] >= dist_threshold_) {
        // Add to the fitness score
        nr++;
      }
    }

    double FP = static_cast<double>(nr) / static_cast<double>(cloud->points.size());

    int percent = static_cast<int>(FP * 100.0);

    return percent;
  }

  int calculate_tp_(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud1, const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud2) {
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>());
    tree->setInputCloud(cloud1);
    std::vector<int> nn_indices(1);
    std::vector<float> nn_dists(1);

    // For each point in the source dataset
    int nr = 0;
    for(size_t i = 0; i < cloud2->points.size(); ++i) {
      // Find its nearest neighbor in the target
      tree_->nearestKSearch(cloud2->points[i], 1, nn_indices, nn_dists);

      // Deal with occlusions (incomplete targets)
      if(nn_dists[0] <= dist_threshold_) {
        // Add to the fitness score
        nr++;
      }
    }

    double TP = static_cast<double>(nr) / static_cast<double>(cloud1->points.size());

    int percent = static_cast<int>(TP * 100.0);

    return percent;
  }

  void load_dump_(bool load_graph) {
    // show the progress modal until loading will be finished
    if(progress_.run("graph load")) {
      auto result = progress_.result<std::shared_ptr<InteractiveGraphView>>();
      if(result == nullptr) {
        pfd::message message("Error", "failed to load graph data", pfd::choice::ok);
        while(!message.ready()) {
          usleep(100);
        }
        return;
      }

      // initialize OpenGL stuffs in this main thread
      result->init_gl();
      update_dump_ = result;
      graph_loaded_ = true;
    }

    if(!load_graph) {
      return;
    }

    pfd::select_folder dialog("choose graph directory");
    while(!dialog.ready()) {
      usleep(100);
    }

    std::string result = dialog.result();
    if(result.empty()) {
      return;
    }

    if(update_dump_->num_vertices() != 0) {
      pfd::message dialog("Confirm", "The current map data will be closed, and unsaved data will be lost.\nDo you want to continue?");
      while(!dialog.ready()) {
        usleep(100);
      }

      if(dialog.result() != pfd::button::ok) {
        return;
      }
    }
    // open the progress modal and load the graph in a background thread
    progress_.open<std::shared_ptr<InteractiveGraphView>>("graph load", [=](guik::ProgressInterface& p) {
      std::shared_ptr<InteractiveGraphView> graph(new InteractiveGraphView());
      if(!graph->load_map_data(result, p)) {
        return std::shared_ptr<InteractiveGraphView>();
      }
      return graph;
    });
  }

};
}  // namespace hdl_graph_slam
