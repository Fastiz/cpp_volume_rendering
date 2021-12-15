#include "../../defines.h"
#include "rc1prenderershadowmap.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/norm.hpp>

#include <vis_utils/camera.h>

#include <volvis_utils/utils.h>
#include <math_utils/utils.h>

#include "imgui.h"
#include "imgui_impl_glut.h"
#include "imgui_impl_opengl2.h"

RayCasting1PassShadowMap::RayCasting1PassShadowMap()
        : m_glsl_transfer_function(nullptr), cp_shader_rendering(nullptr), m_u_step_size(0.5f),
          m_apply_gradient_shading(false) {
#ifdef MULTISAMPLE_AVAILABLE
    vr_pixel_multiscaling_support = true;
#endif
}

RayCasting1PassShadowMap::~RayCasting1PassShadowMap() {
    Clean();
}

void RayCasting1PassShadowMap::Clean() {
    if (m_glsl_transfer_function) delete m_glsl_transfer_function;
    m_glsl_transfer_function = nullptr;

    DestroyRenderingPass();

    BaseVolumeRenderer::Clean();
}

void RayCasting1PassShadowMap::ReloadShaders() {
    cp_shader_rendering->Reload();
    m_rdr_frame_to_screen.ClearShaders();
}

bool RayCasting1PassShadowMap::Init(int swidth, int sheight) {
    if (IsBuilt()) Clean();

    if (m_ext_data_manager->GetCurrentVolumeTexture() == nullptr) return false;
    m_glsl_transfer_function = m_ext_data_manager->GetCurrentTransferFunction()->GenerateTexture_1D_RGBt();

    // Create Rendering Buffers and Shaders
    CreateRenderingPass();
    gl::ExitOnGLError("RayCasting1PassShadowMap: Error on Preparing Models and Shaders");

    // estimate initial integration step
    glm::dvec3 sv = m_ext_data_manager->GetCurrentStructuredVolume()->GetScale();
    m_u_step_size = float((0.5f / glm::sqrt(3.0f)) * glm::sqrt(sv.x * sv.x + sv.y * sv.y + sv.z * sv.z));

    Reshape(swidth, sheight);

    SetBuilt(true);
    SetOutdated();
    return true;
}

bool RayCasting1PassShadowMap::Update(vis::Camera *camera) {

    // RENDERING SHADER ------------------------------------------------------------------------------------------------
    {
        cp_shader_rendering->Bind();

        // MULTISAMPLE
        if (IsPixelMultiScalingSupported() && GetCurrentMultiScalingMode() > 0) {
            cp_shader_rendering->RecomputeNumberOfGroups(m_rdr_frame_to_screen.GetWidth(),
                                                         m_rdr_frame_to_screen.GetHeight(), 0);
        } else {
            cp_shader_rendering->RecomputeNumberOfGroups(m_ext_rendering_parameters->GetScreenWidth(),
                                                         m_ext_rendering_parameters->GetScreenHeight(), 0);
        }

        cp_shader_rendering->SetUniform("CameraEye", camera->GetEye());
        cp_shader_rendering->BindUniform("CameraEye");

        cp_shader_rendering->SetUniform("u_CameraLookAt", camera->LookAt());
        cp_shader_rendering->BindUniform("u_CameraLookAt");

        cp_shader_rendering->SetUniform("ProjectionMatrix", camera->Projection());
        cp_shader_rendering->BindUniform("ProjectionMatrix");

        cp_shader_rendering->SetUniform("u_TanCameraFovY", (float) tan(DEGREE_TO_RADIANS(camera->GetFovY()) / 2.0));
        cp_shader_rendering->BindUniform("u_TanCameraFovY");

        cp_shader_rendering->SetUniform("u_CameraAspectRatio", camera->GetAspectRatio());
        cp_shader_rendering->BindUniform("u_CameraAspectRatio");

        cp_shader_rendering->SetUniform("StepSize", m_u_step_size);
        cp_shader_rendering->BindUniform("StepSize");

        cp_shader_rendering->SetUniform("ApplyOcclusion", 1);
        cp_shader_rendering->BindUniform("ApplyOcclusion");

        cp_shader_rendering->SetUniform("ApplyShadow", 1);
        cp_shader_rendering->BindUniform("ApplyShadow");

        cp_shader_rendering->SetUniform("ApplyGradientPhongShading",
                                        (m_apply_gradient_shading && m_ext_data_manager->GetCurrentGradientTexture())
                                        ? 1
                                        : 0);
        cp_shader_rendering->BindUniform("ApplyGradientPhongShading");

        cp_shader_rendering->SetUniform("BlinnPhongKa", m_ext_rendering_parameters->GetBlinnPhongKambient());
        cp_shader_rendering->BindUniform("BlinnPhongKa");
        cp_shader_rendering->SetUniform("BlinnPhongKd", m_ext_rendering_parameters->GetBlinnPhongKdiffuse());
        cp_shader_rendering->BindUniform("BlinnPhongKd");
        cp_shader_rendering->SetUniform("BlinnPhongKs", m_ext_rendering_parameters->GetBlinnPhongKspecular());
        cp_shader_rendering->BindUniform("BlinnPhongKs");
        cp_shader_rendering->SetUniform("BlinnPhongShininess", m_ext_rendering_parameters->GetBlinnPhongNshininess());
        cp_shader_rendering->BindUniform("BlinnPhongShininess");

        cp_shader_rendering->SetUniform("BlinnPhongIspecular", m_ext_rendering_parameters->GetLightSourceSpecular());
        cp_shader_rendering->BindUniform("BlinnPhongIspecular");

        cp_shader_rendering->SetUniform("WorldEyePos", camera->GetEye());
        cp_shader_rendering->BindUniform("WorldEyePos");

        cp_shader_rendering->SetUniform("LightSourcePosition",
                                        m_ext_rendering_parameters->GetBlinnPhongLightingPosition());
        cp_shader_rendering->BindUniform("LightSourcePosition");

        cp_shader_rendering->BindUniforms();

        gl::Shader::Unbind();
    }
    // -----------------------------------------------------------------------------------------------------------------

    // SHADOW MAP SHADER -----------------------------------------------------------------------------------------------
    {
        cp_shader_shadow_map->Bind();

        // MULTISAMPLE
        if (IsPixelMultiScalingSupported() && GetCurrentMultiScalingMode() > 0) {
            cp_shader_shadow_map->RecomputeNumberOfGroups(m_rdr_frame_to_screen.GetWidth(),
                                                         m_rdr_frame_to_screen.GetHeight(), 0);
        } else {
            cp_shader_shadow_map->RecomputeNumberOfGroups(m_ext_rendering_parameters->GetScreenWidth(),
                                                         m_ext_rendering_parameters->GetScreenHeight(), 0);
        }

        cp_shader_shadow_map->SetUniform("CameraEye", m_ext_rendering_parameters->GetBlinnPhongLightingPosition());
        cp_shader_shadow_map->BindUniform("CameraEye");

        cp_shader_shadow_map->SetUniform(
                "u_CameraLookAt",
                glm::lookAt(
                        m_ext_rendering_parameters->GetBlinnPhongLightingPosition(),
                        m_ext_rendering_parameters->GetBlinnPhongLightingPosition() + m_ext_rendering_parameters->GetBlinnPhongLightSourceCameraForward(),
                        m_ext_rendering_parameters->GetBlinnPhongLightSourceCameraUp()
                        )
                );
        cp_shader_shadow_map->BindUniform("u_CameraLookAt");

        cp_shader_shadow_map->SetUniform("u_TanCameraFovY", (float) tan(DEGREE_TO_RADIANS(camera->GetFovY()) / 2.0));
        cp_shader_shadow_map->BindUniform("u_TanCameraFovY");

        cp_shader_shadow_map->SetUniform("u_CameraAspectRatio", camera->GetAspectRatio());
        cp_shader_shadow_map->BindUniform("u_CameraAspectRatio");

        cp_shader_shadow_map->SetUniform("StepSize", m_u_step_size);
        cp_shader_shadow_map->BindUniform("StepSize");

        cp_shader_shadow_map->BindUniforms();

        gl::Shader::Unbind();
    }
    // -----------------------------------------------------------------------------------------------------------------

    gl::ExitOnGLError("RayCasting1PassShadowMap: After Update.");
    return true;
}

void RayCasting1PassShadowMap::Redraw() {
    m_rdr_frame_to_screen.ClearTexture();

    cp_shader_rendering->Bind();
    m_rdr_frame_to_screen.BindImageTexture();

    cp_shader_rendering->Dispatch();
    gl::ComputeShader::Unbind();

    m_rdr_frame_to_screen.Draw();
}

void RayCasting1PassShadowMap::MultiSampleRedraw() {
    m_rdr_frame_to_screen.ClearTexture();

    cp_shader_rendering->Bind();
    m_rdr_frame_to_screen.BindImageTexture();

    cp_shader_rendering->Dispatch();
    gl::ComputeShader::Unbind();

    m_rdr_frame_to_screen.DrawMultiSampleHigherResolutionMode();
}

void RayCasting1PassShadowMap::DownScalingRedraw() {
    m_rdr_frame_to_screen.ClearTexture();

    cp_shader_rendering->Bind();
    m_rdr_frame_to_screen.BindImageTexture();

    cp_shader_rendering->Dispatch();
    gl::ComputeShader::Unbind();

    m_rdr_frame_to_screen.DrawHigherResolutionWithDownScale();
}

void RayCasting1PassShadowMap::UpScalingRedraw() {
    m_rdr_frame_to_screen.ClearTexture();

    cp_shader_rendering->Bind();
    m_rdr_frame_to_screen.BindImageTexture();

    cp_shader_rendering->Dispatch();
    gl::ComputeShader::Unbind();

    m_rdr_frame_to_screen.DrawLowerResolutionWithUpScale();
}

void RayCasting1PassShadowMap::SetImGuiComponents() {
    ImGui::Separator();
    ImGui::Text("Step Size: ");
    if (ImGui::DragFloat("###RayCasting1PassUIIntegrationStepSize", &m_u_step_size, 0.01f, 0.01f, 100.0f, "%.2f"))
        SetOutdated();

    AddImGuiMultiSampleOptions();

    if (m_ext_data_manager->GetCurrentGradientTexture()) {
        ImGui::Separator();
        if (ImGui::Checkbox("Apply Gradient Shading", &m_apply_gradient_shading)) {
            // Delete current uniform
            cp_shader_rendering->ClearUniform("TexVolumeGradient");

            if (m_apply_gradient_shading && m_ext_data_manager->GetCurrentGradientTexture()) {
                cp_shader_rendering->Bind();
                cp_shader_rendering->SetUniformTexture3D("TexVolumeGradient",
                                                         m_ext_data_manager->GetCurrentGradientTexture()->GetTextureID(),
                                                         3);
                cp_shader_rendering->BindUniform("TexVolumeGradient");
                gl::ComputeShader::Unbind();
            }
            SetOutdated();
        }
        ImGui::Separator();
    }
}

void RayCasting1PassShadowMap::CreateRenderingPass() {
    glm::vec3 vol_resolution = glm::vec3(m_ext_data_manager->GetCurrentStructuredVolume()->GetWidth(),
                                         m_ext_data_manager->GetCurrentStructuredVolume()->GetHeight(),
                                         m_ext_data_manager->GetCurrentStructuredVolume()->GetDepth());

    glm::vec3 vol_voxelsize = glm::vec3(m_ext_data_manager->GetCurrentStructuredVolume()->GetScaleX(),
                                        m_ext_data_manager->GetCurrentStructuredVolume()->GetScaleY(),
                                        m_ext_data_manager->GetCurrentStructuredVolume()->GetScaleZ());

    glm::vec3 vol_aabb = vol_resolution * vol_voxelsize;

    // RENDERING SHADER ------------------------------------------------------------------------------------------------
    {
        cp_shader_rendering = new gl::ComputeShader();
        cp_shader_rendering->AddShaderFile(CPPVOLREND_DIR"structured/_common_shaders/ray_bbox_intersection.comp");
        cp_shader_rendering->AddShaderFile(CPPVOLREND_DIR"structured/rc1shadowmap/ray_marching_1p.comp");
        cp_shader_rendering->LoadAndLink();
        cp_shader_rendering->Bind();

        if (m_ext_data_manager->GetCurrentVolumeTexture())
            cp_shader_rendering->SetUniformTexture3D("TexVolume",
                                                     m_ext_data_manager->GetCurrentVolumeTexture()->GetTextureID(), 1);
        if (m_glsl_transfer_function)
            cp_shader_rendering->SetUniformTexture1D("TexTransferFunc", m_glsl_transfer_function->GetTextureID(), 2);
        if (m_apply_gradient_shading && m_ext_data_manager->GetCurrentGradientTexture())
            cp_shader_rendering->SetUniformTexture3D("TexVolumeGradient",
                                                     m_ext_data_manager->GetCurrentGradientTexture()->GetTextureID(),
                                                     3);

        cp_shader_rendering->SetUniform("VolumeGridResolution", vol_resolution);
        cp_shader_rendering->SetUniform("VolumeVoxelSize", vol_voxelsize);
        cp_shader_rendering->SetUniform("VolumeGridSize", vol_aabb);

        cp_shader_rendering->BindUniforms();
        cp_shader_rendering->Unbind();
    }
    // -----------------------------------------------------------------------------------------------------------------
    
    // SHADOW MAP SHADER -----------------------------------------------------------------------------------------------
    {
        cp_shader_shadow_map = new gl::ComputeShader();
        cp_shader_shadow_map->AddShaderFile(CPPVOLREND_DIR"structured/_common_shaders/ray_bbox_intersection.comp");
        cp_shader_shadow_map->AddShaderFile(CPPVOLREND_DIR"structured/rc1shadowmap/shadow_map.comp");
        cp_shader_shadow_map->LoadAndLink();
        cp_shader_shadow_map->Bind();

        if (m_ext_data_manager->GetCurrentVolumeTexture())
            cp_shader_shadow_map->SetUniformTexture3D("TexVolume",
                                                      m_ext_data_manager->GetCurrentVolumeTexture()->GetTextureID(), 1);
        if (m_glsl_transfer_function)
            cp_shader_shadow_map->SetUniformTexture1D("TexTransferFunc", m_glsl_transfer_function->GetTextureID(), 2);

        cp_shader_shadow_map->SetUniform("VolumeGridResolution", vol_resolution);
        cp_shader_shadow_map->SetUniform("VolumeVoxelSize", vol_voxelsize);
        cp_shader_shadow_map->SetUniform("VolumeGridSize", vol_aabb);

        cp_shader_shadow_map->BindUniforms();
        cp_shader_shadow_map->Unbind();
    }
    // -----------------------------------------------------------------------------------------------------------------
}

void RayCasting1PassShadowMap::DestroyRenderingPass() {
    if (cp_shader_rendering) delete cp_shader_rendering;
    cp_shader_rendering = nullptr;
    if (cp_shader_shadow_map) delete cp_shader_shadow_map;
    cp_shader_shadow_map = nullptr;
    
    gl::ExitOnGLError("Could not destroy shaders");
}

void RayCasting1PassShadowMap::RecreateRenderingPass() {
    DestroyRenderingPass();
    CreateRenderingPass();

    gl::ExitOnGLError("Could not recreate rendering pass");
}
