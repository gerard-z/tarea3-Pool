"""Lighting Shaders"""
"""Shaders personalizados con iluminación, dentro de estos shaders se encuentra los spotliht que sirven para simular
una linterna. """
from OpenGL.GL import *
import OpenGL.GL.shaders
from grafica.gpu_shape import GPUShape, GPUShapeMulti


class SimplePhongDirectionalShaderProgram:

    def __init__(self):
        vertex_shader = """
            #version 330 core

            layout (location = 0) in vec3 position;
            layout (location = 1) in vec3 color;
            layout (location = 2) in vec3 normal;

            out vec3 fragPosition;
            out vec3 fragOriginalColor;
            out vec3 fragNormal;

            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;

            void main()
            {
                fragPosition = vec3(model * vec4(position, 1.0));
                fragOriginalColor = color;
                fragNormal = mat3(transpose(inverse(model))) * normal;  
                
                gl_Position = projection * view * vec4(fragPosition, 1.0);
            }
            """

        fragment_shader = """
            #version 330 core

            out vec4 fragColor;

            in vec3 fragNormal;
            in vec3 fragPosition;
            in vec3 fragOriginalColor;
            
            uniform vec3 lightDirection; 
            uniform vec3 viewPosition;
            uniform vec3 La;
            uniform vec3 Ld;
            uniform vec3 Ls;
            uniform vec3 Ka;
            uniform vec3 Kd;
            uniform vec3 Ks;
            uniform uint shininess;
            uniform float constantAttenuation;
            uniform float linearAttenuation;
            uniform float quadraticAttenuation;

            void main()
            {
                // ambient
                vec3 ambient = Ka * La;
                
                // diffuse
                // fragment normal has been interpolated, so it does not necessarily have norm equal to 1
                vec3 normalizedNormal = normalize(fragNormal);
                vec3 lightDir = normalize(-lightDirection);
                float diff = max(dot(normalizedNormal, lightDir), 0.0);
                vec3 diffuse = Kd * Ld * diff;
                
                // specular
                vec3 viewDir = normalize(viewPosition - fragPosition);
                vec3 reflectDir = reflect(-lightDir, normalizedNormal);  
                float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
                vec3 specular = Ks * Ls * spec;
                    
                vec3 result = (ambient + diffuse + specular ) * fragOriginalColor;
                fragColor = vec4(result, 1.0);
            }
            """

        self.shaderProgram = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(vertex_shader, OpenGL.GL.GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(fragment_shader, OpenGL.GL.GL_FRAGMENT_SHADER))


    def setupVAO(self, gpuShape):

        glBindVertexArray(gpuShape.vao)

        glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)

        # 3d vertices + rgb color + 3d normals => 3*4 + 3*4 + 3*4 = 36 bytes
        position = glGetAttribLocation(self.shaderProgram, "position")
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(0))
        glEnableVertexAttribArray(position)
        
        color = glGetAttribLocation(self.shaderProgram, "color")
        glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(12))
        glEnableVertexAttribArray(color)

        normal = glGetAttribLocation(self.shaderProgram, "normal")
        glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(24))
        glEnableVertexAttribArray(normal)

        # Unbinding current vao
        glBindVertexArray(0)


    def drawCall(self, gpuShape, mode=GL_TRIANGLES):
        assert isinstance(gpuShape, GPUShape)

        # Binding the VAO and executing the draw call
        glBindVertexArray(gpuShape.vao)
        glDrawElements(mode, gpuShape.size, GL_UNSIGNED_INT, None)

        # Unbind the current VAO
        glBindVertexArray(0)


class SimplePhongTextureDirectionalShaderProgram:

    def __init__(self):
        vertex_shader = """
            #version 330 core
            
            in vec3 position;
            in vec2 texCoords;
            in vec3 normal;

            out vec3 fragPosition;
            out vec2 fragTexCoords;
            out vec3 fragNormal;

            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;

            void main()
            {
                fragPosition = vec3(model * vec4(position, 1.0));
                fragTexCoords = texCoords;
                fragNormal = mat3(transpose(inverse(model))) * normal;  
                
                gl_Position = projection * view * vec4(fragPosition, 1.0);
            }
            """

        fragment_shader = """
            #version 330 core

            in vec3 fragNormal;
            in vec3 fragPosition;
            in vec2 fragTexCoords;

            out vec4 fragColor;
            
            uniform vec3 lightDirection; 
            uniform vec3 viewPosition; 
            uniform vec3 La;
            uniform vec3 Ld;
            uniform vec3 Ls;
            uniform vec3 Ka;
            uniform vec3 Kd;
            uniform vec3 Ks;
            uniform uint shininess;
            uniform float constantAttenuation;
            uniform float linearAttenuation;
            uniform float quadraticAttenuation;

            uniform sampler2D samplerTex;

            void main()
            {
                // ambient
                vec3 ambient = Ka * La;
                
                // diffuse
                // fragment normal has been interpolated, so it does not necessarily have norm equal to 1
                vec3 normalizedNormal = normalize(fragNormal);
                vec3 lightDir = normalize(-lightDirection);
                float diff = max(dot(normalizedNormal, lightDir), 0.0);
                vec3 diffuse = Kd * Ld * diff;
                
                // specular
                vec3 viewDir = normalize(viewPosition - fragPosition);
                vec3 reflectDir = reflect(-lightDir, normalizedNormal);  
                float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
                vec3 specular = Ks * Ls * spec;
                    
                vec4 fragOriginalColor = texture(samplerTex, fragTexCoords);

                vec3 result = (ambient + diffuse + specular ) * fragOriginalColor.rgb;
                fragColor = vec4(result, 1.0);
            }
            """

        self.shaderProgram = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(vertex_shader, OpenGL.GL.GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(fragment_shader, OpenGL.GL.GL_FRAGMENT_SHADER))


    def setupVAO(self, gpuShape):

        glBindVertexArray(gpuShape.vao)

        glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)

        # 3d vertices + rgb color + 3d normals => 3*4 + 2*4 + 3*4 = 32 bytes
        position = glGetAttribLocation(self.shaderProgram, "position")
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        glEnableVertexAttribArray(position)
        
        color = glGetAttribLocation(self.shaderProgram, "texCoords")
        glVertexAttribPointer(color, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        glEnableVertexAttribArray(color)

        normal = glGetAttribLocation(self.shaderProgram, "normal")
        glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))
        glEnableVertexAttribArray(normal)

        # Unbinding current vao
        glBindVertexArray(0)


    def drawCall(self, gpuShape, mode=GL_TRIANGLES):
        assert isinstance(gpuShape, GPUShape)

        # Binding the VAO and executing the draw call
        glBindVertexArray(gpuShape.vao)
        glBindTexture(GL_TEXTURE_2D, gpuShape.texture)

        glDrawElements(mode, gpuShape.size, GL_UNSIGNED_INT, None)

        # Unbind the current VAO
        glBindVertexArray(0)


class MultiplePhongShaderProgram:

    def __init__(self):
        vertex_shader = """
            #version 330 core

            layout (location = 0) in vec3 position;
            layout (location = 1) in vec3 color;
            layout (location = 2) in vec3 normal;

            out vec3 fragPosition;
            out vec3 fragOriginalColor;
            out vec3 fragNormal;

            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;

            void main()
            {
                fragPosition = vec3(model * vec4(position, 1.0));
                fragOriginalColor = color;
                fragNormal = mat3(transpose(inverse(model))) * normal;  
                
                gl_Position = projection * view * vec4(fragPosition, 1.0);
            }
            """

        fragment_shader = """
            #version 330 core

            out vec4 fragColor;

            in vec3 fragNormal;
            in vec3 fragPosition;
            in vec3 fragOriginalColor;
            
            uniform vec3 lightPos0; 
            uniform vec3 lightPos1;  
            uniform vec3 lightPos2;  
            uniform vec3 viewPosition;
            uniform vec3 La0;
            uniform vec3 La1;
            uniform vec3 La2;
            uniform vec3 Ld0;
            uniform vec3 Ld1;
            uniform vec3 Ld2;
            uniform vec3 Ls0;
            uniform vec3 Ls1;
            uniform vec3 Ls2;
            uniform vec3 Ka;
            uniform vec3 Kd;
            uniform vec3 Ks;
            uniform uint shininess;
            uniform float constantAttenuation;
            uniform float linearAttenuation;
            uniform float quadraticAttenuation;

            void main()
            {
                // fragment normal has been interpolated, so it does not necessarily have norm equal to 1
                vec3 normalizedNormal = normalize(fragNormal);
                vec3 result = vec3(0.0f, 0.0f, 0.0f);

                vec3 lights[3] = vec3[](lightPos0, lightPos1, lightPos2);
                vec3 La[3] = vec3[](La0, La1, La2);
                vec3 Ld[3] = vec3[](Ld0, Ld1, Ld2);
                vec3 Ls[3] = vec3[](Ls0, Ls1, Ls2);

                for (int i = 0; i < 3; i++)
                {
                    // ambient
                    vec3 ambient = Ka * La[i];

                    // diffuse
                    vec3 toLight = lights[i] - fragPosition;
                    vec3 lightDir = normalize(toLight);
                    float diff = max(dot(normalizedNormal, lightDir), 0.0);
                    vec3 diffuse = Kd * Ld[i] * diff;
                    
                    // specular
                    vec3 viewDir = normalize(viewPosition - fragPosition);
                    vec3 reflectDir = reflect(-lightDir, normalizedNormal);  
                    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
                    vec3 specular = Ks * Ls[i] * spec;

                    // attenuation
                    float distToLight = length(toLight);
                    float attenuation = constantAttenuation
                        + linearAttenuation * distToLight
                        + quadraticAttenuation * distToLight * distToLight;
                        
                    result += ambient +  ((diffuse + specular) / attenuation) ;
                }

                result = result * fragOriginalColor;
                fragColor = vec4(result, 1.0);
            }
            """

        self.shaderProgram = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(vertex_shader, OpenGL.GL.GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(fragment_shader, OpenGL.GL.GL_FRAGMENT_SHADER))


    def setupVAO(self, gpuShape):

        glBindVertexArray(gpuShape.vao)

        glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)

        # 3d vertices + rgb color + 3d normals => 3*4 + 3*4 + 3*4 = 36 bytes
        position = glGetAttribLocation(self.shaderProgram, "position")
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(0))
        glEnableVertexAttribArray(position)
        
        color = glGetAttribLocation(self.shaderProgram, "color")
        glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(12))
        glEnableVertexAttribArray(color)

        normal = glGetAttribLocation(self.shaderProgram, "normal")
        glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(24))
        glEnableVertexAttribArray(normal)

        # Unbinding current vao
        glBindVertexArray(0)


    def drawCall(self, gpuShape, mode=GL_TRIANGLES):
        assert isinstance(gpuShape, GPUShape)

        # Binding the VAO and executing the draw call
        glBindVertexArray(gpuShape.vao)
        glDrawElements(mode, gpuShape.size, GL_UNSIGNED_INT, None)

        # Unbind the current VAO
        glBindVertexArray(0)


class MultipleTexturePhongShaderProgram:

    def __init__(self):
        vertex_shader = """
            #version 330 core
            
            in vec3 position;
            in vec2 texCoords;
            in vec3 normal;

            out vec3 fragPosition;
            out vec2 fragTexCoords;
            out vec3 fragNormal;

            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;

            void main()
            {
                fragPosition = vec3(model * vec4(position, 1.0));
                fragTexCoords = texCoords;
                fragNormal = mat3(transpose(inverse(model))) * normal;  
                
                gl_Position = projection * view * vec4(fragPosition, 1.0);
            }
            """

        fragment_shader = """
            #version 330 core

            in vec3 fragNormal;
            in vec3 fragPosition;
            in vec2 fragTexCoords;

            out vec4 fragColor;
            
            uniform vec3 lightPos0; 
            uniform vec3 lightPos1;  
            uniform vec3 lightPos2;  
            uniform vec3 viewPosition; 
            uniform vec3 La0;
            uniform vec3 La1;
            uniform vec3 La2;
            uniform vec3 Ld0;
            uniform vec3 Ld1;
            uniform vec3 Ld2;
            uniform vec3 Ls0;
            uniform vec3 Ls1;
            uniform vec3 Ls2;
            uniform vec3 Ka;
            uniform vec3 Kd;
            uniform vec3 Ks;
            uniform uint shininess;
            uniform float constantAttenuation;
            uniform float linearAttenuation;
            uniform float quadraticAttenuation;

            uniform sampler2D samplerTex;

            void main()
            {
                // fragment normal has been interpolated, so it does not necessarily have norm equal to 1
                vec3 normalizedNormal = normalize(fragNormal);
                vec4 fragOriginalColor = texture(samplerTex, fragTexCoords);
                vec3 result = vec3(0.0f, 0.0f, 0.0f);

                vec3 lights[3] = vec3[](lightPos0, lightPos1, lightPos2);
                vec3 La[3] = vec3[](La0, La1, La2);
                vec3 Ld[3] = vec3[](Ld0, Ld1, Ld2);
                vec3 Ls[3] = vec3[](Ls0, Ls1, Ls2);

                for (int i = 0; i < 3; i++)
                {
                    // ambient
                    vec3 ambient = Ka * La[i];

                    // diffuse
                    vec3 toLight = lights[i] - fragPosition;
                    vec3 lightDir = normalize(toLight);
                    float diff = max(dot(normalizedNormal, lightDir), 0.0);
                    vec3 diffuse = Kd * Ld[i] * diff;
                    
                    // specular
                    vec3 viewDir = normalize(viewPosition - fragPosition);
                    vec3 reflectDir = reflect(-lightDir, normalizedNormal);  
                    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
                    vec3 specular = Ks * Ls[i] * spec;

                    // attenuation
                    float distToLight = length(toLight);
                    float attenuation = constantAttenuation
                        + linearAttenuation * distToLight
                        + quadraticAttenuation * distToLight * distToLight;
                        
                    result += ambient + ((diffuse + specular) / attenuation) ;
                }

                result = result * fragOriginalColor.rgb;
                fragColor = vec4(result, 1.0);
            }
            """

        self.shaderProgram = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(vertex_shader, OpenGL.GL.GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(fragment_shader, OpenGL.GL.GL_FRAGMENT_SHADER))


    def setupVAO(self, gpuShape):

        glBindVertexArray(gpuShape.vao)

        glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)

        # 3d vertices + rgb color + 3d normals => 3*4 + 2*4 + 3*4 = 32 bytes
        position = glGetAttribLocation(self.shaderProgram, "position")
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        glEnableVertexAttribArray(position)
        
        color = glGetAttribLocation(self.shaderProgram, "texCoords")
        glVertexAttribPointer(color, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        glEnableVertexAttribArray(color)

        normal = glGetAttribLocation(self.shaderProgram, "normal")
        glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))
        glEnableVertexAttribArray(normal)

        # Unbinding current vao
        glBindVertexArray(0)


    def drawCall(self, gpuShape, mode=GL_TRIANGLES):
        assert isinstance(gpuShape, GPUShape)

        # Binding the VAO and executing the draw call
        glBindVertexArray(gpuShape.vao)
        glBindTexture(GL_TEXTURE_2D, gpuShape.texture)

        glDrawElements(mode, gpuShape.size, GL_UNSIGNED_INT, None)

        # Unbind the current VAO
        glBindVertexArray(0)

####### Shaders para una luz spotlight (Linterna)


class SimplePhongSpotlightShaderProgram:

    def __init__(self):
        vertex_shader = """
            #version 330 core

            layout (location = 0) in vec3 position;
            layout (location = 1) in vec3 color;
            layout (location = 2) in vec3 normal;

            out vec3 fragPosition;
            out vec3 fragOriginalColor;
            out vec3 fragNormal;

            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;

            void main()
            {
                fragPosition = vec3(model * vec4(position, 1.0));
                fragOriginalColor = color;
                fragNormal = mat3(transpose(inverse(model))) * normal;  
                
                gl_Position = projection * view * vec4(fragPosition, 1.0);
            }
            """

        fragment_shader = """
            #version 330 core

            out vec4 fragColor;

            in vec3 fragNormal;
            in vec3 fragPosition;
            in vec3 fragOriginalColor;
            
            // Iluminación de la linterna
            uniform vec3 lightPos;
            uniform vec3 lightDirection; 
            uniform vec3 viewPosition;
            uniform vec3 La;
            uniform vec3 Ld;
            uniform vec3 Ls;
            uniform vec3 Ka;
            uniform vec3 Kd;
            uniform vec3 Ks;
            uniform uint shininess;
            uniform uint concentration;
            uniform float constantAttenuation;
            uniform float linearAttenuation;
            uniform float quadraticAttenuation;

            // Iluminación de focos
            uniform vec3 lightPos0; 
            uniform vec3 lightPos1;  
            uniform vec3 lightPos2;
            uniform vec3 lightPos3;
            uniform vec3 La0;
            uniform vec3 La1;
            uniform vec3 La2;
            uniform vec3 La3;
            uniform vec3 Ld0;
            uniform vec3 Ld1;
            uniform vec3 Ld2;
            uniform vec3 Ld3;
            uniform vec3 Ls0;
            uniform vec3 Ls1;
            uniform vec3 Ls2;
            uniform vec3 Ls3;

            void main() //Linea 43
            {   
                // Creación de parámetros para múltiples luces
                // fragment normal has been interpolated, so it does not necessarily have norm equal to 1
                vec3 normalizedNormal = normalize(fragNormal);
                vec3 result = vec3(0.0f, 0.0f, 0.0f);

                // Luces
                vec3 lights[3] = vec3[](lightPos0, lightPos1, lightPos2);
                vec3 Las[3] = vec3[](La0, La1, La2);
                vec3 Lds[3] = vec3[](Ld0, Ld1, Ld2);
                vec3 Lss[3] = vec3[](Ls0, Ls1, Ls2);

                for (int i = 0; i<3; i++)   //Linea 56
                {
                    // ambient       Line 58
                    vec3 ambient = Ka * Las[i];

                    // diffuse       Line 61
                    vec3 tolight = lights[i]-fragPosition;
                    vec3 lightDir = normalize(tolight);
                    float diff = max(dot(normalizedNormal, lightDir), 0.0);
                    vec3 diffuse = Kd * Lds[i] * diff;
                
                    // specular     line 67
                    vec3 viewDir = normalize(viewPosition - fragPosition);
                    vec3 reflectDir = reflect(-lightDir, normalizedNormal);  
                    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
                    vec3 specular = Ks * Lss[i] * spec;

                    // attenuation  Line 73
                    float distToLight = length(tolight);
                    float attenuation = 0.01
                        + 0.02 * distToLight
                        + 0.03 * distToLight * distToLight;

                    result += ambient + (((diffuse + specular) / attenuation));
                }
                // Luz del final    Line 81
                // ambient
                vec3 ambient = Ka * La3;

                // diffuse 
                vec3 tolight = lightPos3-fragPosition;
                vec3 lightDir = normalize(tolight);
                float diff = max(dot(normalizedNormal, lightDir), 0.0);
                vec3 diffuse = Kd * Ld3 * diff;
            
                // specular 
                vec3 viewDir = normalize(viewPosition - fragPosition);
                vec3 reflectDir = reflect(-lightDir, normalizedNormal);  
                float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
                vec3 specular = Ks * Ls3 * spec;

                // attenuation
                float distToLight = length(tolight);
                float attenuation = 0.01
                    + 0.01 * distToLight
                    + 0.01 * distToLight * distToLight;

                result += ambient + (((diffuse + specular) / attenuation));

                // Linterna
                // ambient
                ambient = Ka * La;

                // diffuse 
                tolight = lightPos-fragPosition;
                lightDir = normalize(tolight);
                diff = max(dot(normalizedNormal, lightDir), 0.0);
                diffuse = Kd * Ld * diff;
            
                // specular 
                viewDir = normalize(viewPosition - fragPosition);
                reflectDir = reflect(-lightDir, normalizedNormal);  
                spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
                specular = Ks * Ls * spec;

                // attenuation
                distToLight = length(tolight);
                attenuation = constantAttenuation
                    + linearAttenuation * distToLight
                    + quadraticAttenuation * distToLight * distToLight;

                // spotlight 
                vec3 dir = normalize(-lightDirection);
                float spotLight = pow(max(dot(dir, lightDir), 0.0), concentration);

                result += ambient + (spotLight * ((diffuse + specular) / attenuation));

                vec3 resultFin = result * fragOriginalColor;
                fragColor = vec4(resultFin, 1.0);
            }
            """

        self.shaderProgram = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(vertex_shader, OpenGL.GL.GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(fragment_shader, OpenGL.GL.GL_FRAGMENT_SHADER))


    def setupVAO(self, gpuShape):

        glBindVertexArray(gpuShape.vao)

        glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)

        # 3d vertices + rgb color + 3d normals => 3*4 + 3*4 + 3*4 = 36 bytes
        position = glGetAttribLocation(self.shaderProgram, "position")
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(0))
        glEnableVertexAttribArray(position)
        
        color = glGetAttribLocation(self.shaderProgram, "color")
        glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(12))
        glEnableVertexAttribArray(color)

        normal = glGetAttribLocation(self.shaderProgram, "normal")
        glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, 36, ctypes.c_void_p(24))
        glEnableVertexAttribArray(normal)

        # Unbinding current vao
        glBindVertexArray(0)


    def drawCall(self, gpuShape, mode=GL_TRIANGLES):
        assert isinstance(gpuShape, GPUShape)

        # Binding the VAO and executing the draw call
        glBindVertexArray(gpuShape.vao)
        glDrawElements(mode, gpuShape.size, GL_UNSIGNED_INT, None)

        # Unbind the current VAO
        glBindVertexArray(0)


class DoublePhongTextureSpotlightShaderProgram:

    def __init__(self):
        vertex_shader = """
            #version 330 core
            
            in vec3 position;
            in vec2 texCoords;
            in vec3 normal;

            out vec3 fragPosition;
            out vec2 fragTexCoords;
            out vec3 fragNormal;

            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;

            void main()
            {
                fragPosition = vec3(model * vec4(position, 1.0));
                fragTexCoords = texCoords;
                fragNormal = mat3(transpose(inverse(model))) * normal;  
                
                gl_Position = projection * view * vec4(fragPosition, 1.0);
            }
            """

        fragment_shader = """
            #version 330 core

            in vec3 fragNormal;
            in vec3 fragPosition;
            in vec2 fragTexCoords;

            out vec4 fragColor;
            
            // Linterna
            uniform vec3 lightPos;
            uniform vec3 lightDirection; 
            uniform vec3 viewPosition; 
            uniform vec3 La;
            uniform vec3 Ld;
            uniform vec3 Ls;
            uniform vec3 Ka;
            uniform vec3 Kd;
            uniform vec3 Ks;
            uniform uint concentration;
            uniform uint shininess;
            uniform float constantAttenuation;
            uniform float linearAttenuation;
            uniform float quadraticAttenuation;

            // Iluminación de focos
            uniform vec3 lightPos0; 
            uniform vec3 lightPos1;  
            uniform vec3 lightPos2;
            uniform vec3 lightPos3;
            uniform vec3 La0;
            uniform vec3 La1;
            uniform vec3 La2;
            uniform vec3 La3;
            uniform vec3 Ld0;
            uniform vec3 Ld1;
            uniform vec3 Ld2;
            uniform vec3 Ld3;
            uniform vec3 Ls0;
            uniform vec3 Ls1;
            uniform vec3 Ls2;
            uniform vec3 Ls3;
            
            // Información de texturas
            uniform float time;

            uniform sampler2D TexWater;
            uniform sampler2D TexDisplacement;

            void main()
            {    
                // Water effects
                vec4 originalColor;
                vec2 displaCoords= vec2(mod((fragTexCoords.x*0.8+0.1) + time/3, 1), mod((fragTexCoords.y*0.8+0.1) - time/3, 1));
                vec4 displacement = texture(TexDisplacement, displaCoords);
                float desplazamiento = dot(displacement, vec4(1,1,1,1))/20;
                vec2 TexCoords= vec2(mod(fragTexCoords.x + desplazamiento- time, 1), mod(fragTexCoords.y +desplazamiento, 1));
                vec4 waterColor = texture(TexWater, TexCoords);
                originalColor = waterColor;

                // Creación de parámetros para múltiples luces
                // fragment normal has been interpolated, so it does not necessarily have norm equal to 1
                vec3 normalizedNormal = normalize(fragNormal);
                vec3 result = vec3(0.0f, 0.0f, 0.0f);

                // Luces
                vec3 lights[3] = vec3[](lightPos0, lightPos1, lightPos2);
                vec3 Las[3] = vec3[](La0, La1, La2);
                vec3 Lds[3] = vec3[](Ld0, Ld1, Ld2);
                vec3 Lss[3] = vec3[](Ls0, Ls1, Ls2);

                for (int i = 0; i<3; i++)   
                {
                    // ambient      
                    vec3 ambient = Ka * Las[i];

                    // diffuse      
                    vec3 tolight = lights[i]-fragPosition;
                    vec3 lightDir = normalize(tolight);
                    float diff = max(dot(normalizedNormal, lightDir), 0.0);
                    vec3 diffuse = Kd * Lds[i] * diff;
                
                    // specular    
                    vec3 viewDir = normalize(viewPosition - fragPosition);
                    vec3 reflectDir = reflect(-lightDir, normalizedNormal);  
                    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
                    vec3 specular = Ks * Lss[i] * spec;

                    // attenuation  
                    float distToLight = length(tolight);
                    float attenuation = 0.01
                        + 0.02 * distToLight
                        + 0.03 * distToLight * distToLight;

                    result += ambient + (((diffuse + specular) / attenuation));
                }
                // Luz del final  
                // ambient
                vec3 ambient = Ka * La3;

                // diffuse 
                vec3 tolight = lightPos3-fragPosition;
                vec3 lightDir = normalize(tolight);
                float diff = max(dot(normalizedNormal, lightDir), 0.0);
                vec3 diffuse = Kd * Ld3 * diff;
            
                // specular 
                vec3 viewDir = normalize(viewPosition - fragPosition);
                vec3 reflectDir = reflect(-lightDir, normalizedNormal);  
                float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
                vec3 specular = Ks * Ls3 * spec;

                // attenuation
                float distToLight = length(tolight);
                float attenuation = 0.01
                    + 0.01 * distToLight
                    + 0.01 * distToLight * distToLight;

                result += ambient + (((diffuse + specular) / attenuation));

                // Linterna
                // ambient
                ambient = Ka * La;

                // diffuse 
                tolight = lightPos-fragPosition;
                lightDir = normalize(tolight);
                diff = max(dot(normalizedNormal, lightDir), 0.0);
                diffuse = Kd * Ld * diff;
            
                // specular 
                viewDir = normalize(viewPosition - fragPosition);
                reflectDir = reflect(-lightDir, normalizedNormal);  
                spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
                specular = Ks * Ls * spec;

                // attenuation
                distToLight = length(tolight);
                attenuation = constantAttenuation
                    + linearAttenuation * distToLight
                    + quadraticAttenuation * distToLight * distToLight;

                // spotlight 
                vec3 dir = normalize(-lightDirection);
                float spotLight = pow(max(dot(dir, lightDir), 0.0), concentration);

                result += ambient + (spotLight * ((diffuse + specular) / attenuation));
                    
                vec3 resultFin = result * originalColor.rgb;
                fragColor = vec4(resultFin, 1.0);

            }
            """

        self.shaderProgram = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(vertex_shader, OpenGL.GL.GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(fragment_shader, OpenGL.GL.GL_FRAGMENT_SHADER))


    def setupVAO(self, gpuShape):

        glBindVertexArray(gpuShape.vao)

        glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)

        # 3d vertices + rgb color + 3d normals => 3*4 + 2*4 + 3*4 = 32 bytes
        position = glGetAttribLocation(self.shaderProgram, "position")
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        glEnableVertexAttribArray(position)
        
        color = glGetAttribLocation(self.shaderProgram, "texCoords")
        glVertexAttribPointer(color, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        glEnableVertexAttribArray(color)

        normal = glGetAttribLocation(self.shaderProgram, "normal")
        glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))
        glEnableVertexAttribArray(normal)

        # Unbinding current vao
        glBindVertexArray(0)


    def drawCall(self, gpuShape, mode=GL_TRIANGLES):
        assert isinstance(gpuShape, GPUShapeMulti)
        cantidad = gpuShape.cantidad

        # Binding the VAO and executing the draw call
        glBindVertexArray(gpuShape.vao)
        for i in range(cantidad):
            glActiveTexture(GL_TEXTURE0 + i)
            glBindTexture(GL_TEXTURE_2D, gpuShape.texture[i])

        glDrawElements(mode, gpuShape.size, GL_UNSIGNED_INT, None)

        # Unbind the current VAO
        glBindVertexArray(0)


class NormalPhongTextureSpotlightShaderProgram:

    def __init__(self):
        vertex_shader = """
            #version 330 core
            
            in vec3 position;
            in vec2 texCoords;
            //in vec3 normal;

            out vec3 fragPosition;
            out vec2 fragTexCoords;
            //out vec3 fragNormal;

            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;

            void main()
            {
                fragPosition = vec3(model * vec4(position, 1.0));
                fragTexCoords = texCoords;
                //fragNormal = mat3(transpose(inverse(model))) * normal;  
                
                gl_Position = projection * view * vec4(fragPosition, 1.0);
            }
            """

        fragment_shader = """
            #version 330 core

            //in vec3 fragNormal;
            in vec3 fragPosition;
            in vec2 fragTexCoords;

            out vec4 fragColor;

            uniform mat4 model;
            
            // Linterna
            uniform vec3 lightPos;
            uniform vec3 lightDirection; 
            uniform vec3 viewPosition; 
            uniform vec3 La;
            uniform vec3 Ld;
            uniform vec3 Ls;
            uniform vec3 Ka;
            uniform vec3 Kd;
            uniform vec3 Ks;
            uniform uint concentration;
            uniform uint shininess;
            uniform float constantAttenuation;
            uniform float linearAttenuation;
            uniform float quadraticAttenuation;

            // Iluminación de focos
            uniform vec3 lightPos0; 
            uniform vec3 lightPos1;  
            uniform vec3 lightPos2;
            uniform vec3 lightPos3;
            uniform vec3 La0;
            uniform vec3 La1;
            uniform vec3 La2;
            uniform vec3 La3;
            uniform vec3 Ld0;
            uniform vec3 Ld1;
            uniform vec3 Ld2;
            uniform vec3 Ld3;
            uniform vec3 Ls0;
            uniform vec3 Ls1;
            uniform vec3 Ls2;
            uniform vec3 Ls3;

            uniform sampler2D Texture;
            uniform sampler2D normalTexture;

            void main()
            {    
                // Water effects
                vec4 originalColor;
                vec4 normal = texture(normalTexture, fragTexCoords);
                vec4 texColor = texture(Texture, fragTexCoords);
                originalColor = texColor;

                // Creación de parámetros para múltiples luces
                normal = mat4(transpose(inverse(model))) * normal;
                // fragment normal has been interpolated, so it does not necessarily have norm equal to 1
                vec3 normalizedNormal = normalize(normal.xyz);
                vec3 result = vec3(0.0f, 0.0f, 0.0f);

                // Luces
                vec3 lights[3] = vec3[](lightPos0, lightPos1, lightPos2);
                vec3 Las[3] = vec3[](La0, La1, La2);
                vec3 Lds[3] = vec3[](Ld0, Ld1, Ld2);
                vec3 Lss[3] = vec3[](Ls0, Ls1, Ls2);

                for (int i = 0; i<3; i++)   
                {
                    // ambient      
                    vec3 ambient = Ka * Las[i];

                    // diffuse      
                    vec3 tolight = lights[i]-fragPosition;
                    vec3 lightDir = normalize(tolight);
                    float diff = max(dot(normalizedNormal, lightDir), 0.0);
                    vec3 diffuse = Kd * Lds[i] * diff;
                
                    // specular    
                    vec3 viewDir = normalize(viewPosition - fragPosition);
                    vec3 reflectDir = reflect(-lightDir, normalizedNormal);  
                    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
                    vec3 specular = Ks * Lss[i] * spec;

                    // attenuation  
                    float distToLight = length(tolight);
                    float attenuation = 0.01
                        + 0.02 * distToLight
                        + 0.03 * distToLight * distToLight;

                    result += ambient + (((diffuse + specular) / attenuation));
                }
                // Luz del final  
                // ambient
                vec3 ambient = Ka * La3;

                // diffuse 
                vec3 tolight = lightPos3-fragPosition;
                vec3 lightDir = normalize(tolight);
                float diff = max(dot(normalizedNormal, lightDir), 0.0);
                vec3 diffuse = Kd * Ld3 * diff;
            
                // specular 
                vec3 viewDir = normalize(viewPosition - fragPosition);
                vec3 reflectDir = reflect(-lightDir, normalizedNormal);  
                float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
                vec3 specular = Ks * Ls3 * spec;

                // attenuation
                float distToLight = length(tolight);
                float attenuation = 0.01
                    + 0.01 * distToLight
                    + 0.01 * distToLight * distToLight;

                result += ambient + (((diffuse + specular) / attenuation));

                // Linterna
                // ambient
                ambient = Ka * La;

                // diffuse 
                tolight = lightPos-fragPosition;
                lightDir = normalize(tolight);
                diff = max(dot(normalizedNormal, lightDir), 0.0);
                diffuse = Kd * Ld * diff;
            
                // specular 
                viewDir = normalize(viewPosition - fragPosition);
                reflectDir = reflect(-lightDir, normalizedNormal);  
                spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
                specular = Ks * Ls * spec;

                // attenuation
                distToLight = length(tolight);
                attenuation = constantAttenuation
                    + linearAttenuation * distToLight
                    + quadraticAttenuation * distToLight * distToLight;

                // spotlight 
                vec3 dir = normalize(-lightDirection);
                float spotLight = pow(max(dot(dir, lightDir), 0.0), concentration);

                result += ambient + (spotLight * ((diffuse + specular) / attenuation));
                    
                vec3 resultFin = result * originalColor.rgb;
                fragColor = vec4(resultFin, 1.0);

            }
            """

        self.shaderProgram = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(vertex_shader, OpenGL.GL.GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(fragment_shader, OpenGL.GL.GL_FRAGMENT_SHADER))


    def setupVAO(self, gpuShape):

        glBindVertexArray(gpuShape.vao)

        glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)

        # 3d vertices + rgb color + 3d normals => 3*4 + 2*4 = 20 bytes
        position = glGetAttribLocation(self.shaderProgram, "position")
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))
        glEnableVertexAttribArray(position)
        
        color = glGetAttribLocation(self.shaderProgram, "texCoords")
        glVertexAttribPointer(color, 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))
        glEnableVertexAttribArray(color)

        # Unbinding current vao
        glBindVertexArray(0)


    def drawCall(self, gpuShape, mode=GL_TRIANGLES):
        assert isinstance(gpuShape, GPUShapeMulti)
        cantidad = gpuShape.cantidad

        # Binding the VAO and executing the draw call
        glBindVertexArray(gpuShape.vao)
        for i in range(cantidad):
            glActiveTexture(GL_TEXTURE0 + i)
            glBindTexture(GL_TEXTURE_2D, gpuShape.texture[i])

        glDrawElements(mode, gpuShape.size, GL_UNSIGNED_INT, None)

        # Unbind the current VAO
        glBindVertexArray(0)