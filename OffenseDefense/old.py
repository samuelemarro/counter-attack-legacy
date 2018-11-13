def main():
    trainloader, testloader = model_tools.get_cifar10_loaders(1, 1, 1)

    model = prepare_model()

    #optimizer = optim.SGD(surrogate.parameters(), lr=0.1, momentum=0.9,
    #weight_decay=1e-4)

    #model.cpu()#model.cuda()
    model.eval()

    foolbox_model = foolbox.models.PyTorchModel(model, (0, 1), 10, channel_axis=3, device=torch.cuda.current_device(), preprocessing=(0,1))

    cifar_names = [
        'Plane',
        'Car',
        'Bird',
        'Cat',
        'Deer',
        'Dog',
        'Frog',
        'Horse',
        'Ship',
        'Truck'
        ]

    average_anti_genuine = utils.AverageMeter()
    average_anti_adversarial = utils.AverageMeter()
    average_adversarial = utils.AverageMeter()

    #Temp
    average_success_meter = utils.AverageMeter()

    anti_genuine_distances = []
    anti_adversarial_distances = []
    
    plt.ion()
    for data in testloader:
        image, label = data
        image = image[0].numpy()
        label = label[0].numpy()
        #print(image)
        #print(image.shape)
        #show_image(image)
        print('Running')
        
        predicted_label = np.argmax(foolbox_model.predictions(image), 0)
        if predicted_label == label:
            adversarial_attack_criterion = foolbox.criteria.ConfidentMisclassification(0.99) #foolbox.criteria.Misclassification()#foolbox.criteria.ConfidentMisclassification(0.75)
            adversarial_attack = foolbox.attacks.GradientAttack(foolbox_model, adversarial_attack_criterion) #ExpandedDeepFool(foolbox_model, adversarial_attack_criterion, minimum_anti_distance=4.5959e-7)

            anti_attack_criterion = foolbox.criteria.Misclassification()
            anti_attack = foolbox.attacks.GradientAttack(foolbox_model, anti_attack_criterion)

            #adversarial_attack = foolbox.attacks.DeepFoolAttack(foolbox_model, adversarial_attack_criterion) if np.random.rand() > 0.5 else foolbox.attacks.GradientSignAttack(foolbox_model, adversarial_attack_criterion)
            
            #minimum_distance = 4.5959e-7
            try:
                #detector = detectors.AntiAdversarialDistance(anti_attack, minimum_distance)
                #detection_aware_image = detectors.DetectionAwareAdversarial(foolbox_model, adversarial_attack_criterion, image, label, detector)
                #adversarial = adversarial_attack(detection_aware_image)
                adversarial = adversarial_attack(image, label)
            except Exception:
                print('Unknown Attack Error')
                traceback.print_exc()
                adversarial = None
            if adversarial is not None:
                adversarial_label = np.argmax(foolbox_model.predictions(adversarial), 0)

                try:
                    anti_adversarial = anti_attack(adversarial, adversarial_label)
                except:
                    print('Unknown Anti-Adversarial Error')
                    traceback.print_exc()
                    anti_adversarial = None

                try:
                    anti_genuine = anti_attack(image, label)
                except:
                    print('Unknown Anti-Genuine Error')
                    traceback.print_exc()
                    anti_genuine = None

                if anti_adversarial is None:
                    print('Anti-Adversarial failed')
                elif anti_genuine is None:
                    print('Anti-Genuine failed')
                else:
                    anti_adversarial_label = np.argmax(foolbox_model.predictions(anti_adversarial), 0)
                    anti_genuine_label = np.argmax(foolbox_model.predictions(anti_genuine), 0)

                    adversarial_distance = np.average(np.power((adversarial - image), 2))
                    anti_adversarial_distance = np.average(np.power((adversarial - anti_adversarial), 2))
                    anti_genuine_distance = np.average(np.power((image - anti_genuine), 2))

                    #Temp
                    print('Entering Temp')
                    def softmax(x):
                        """Compute softmax values for each sets of scores in x."""
                        e_x = np.exp(x - np.max(x))
                        return e_x / e_x.sum()
                    genuine_confidence = np.max(softmax(foolbox_model.predictions(image)))
                    print('Genuine Confidence: {}'.format(genuine_confidence))
                    adversarial_confidence = np.max(softmax(foolbox_model.predictions(adversarial)))
                    print('Adversarial Confidence: {}'.format(adversarial_confidence))

                    hyper_adversarial_genuine = foolbox.attacks.GradientAttack(foolbox_model, foolbox.criteria.TargetClassProbability(anti_genuine_label, 0.9))(anti_genuine, anti_genuine_label)
                    hyper_adversarial_adversarial = foolbox.attacks.GradientAttack(foolbox_model, foolbox.criteria.TargetClassProbability(anti_adversarial_label, 0.9))(anti_adversarial, anti_adversarial_label)

                    if hyper_adversarial_genuine is None and hyper_adversarial_adversarial is None:
                        print('Failed both')
                        continue
                    elif hyper_adversarial_genuine is None:
                        average_success_meter.update(1)
                        print('Accuracy: {:2.2f}%'.format(average_success_meter.avg * 100.0))
                        continue
                    elif hyper_adversarial_adversarial is None:
                        average_success_meter.update(0)
                        print('Accuracy: {:2.2f}%'.format(average_success_meter.avg * 100.0))
                        continue

                    hyper_distance_genuine = np.average(np.power((hyper_adversarial_genuine - anti_genuine), 2))
                    hyper_distance_adversarial = np.average(np.power((hyper_adversarial_adversarial - anti_adversarial), 2))
                    
                    genuine_prediction = hyper_distance_genuine > anti_genuine_distance
                    adversarial_prediction = hyper_distance_adversarial > anti_adversarial_distance

                    average_success_meter.update(0 if genuine_prediction else 1)
                    average_success_meter.update(1 if adversarial_prediction else 0)
                    print('Accuracy: {:2.2f}%'.format(average_success_meter.avg * 100.0))
                    #End Temp

                    average_anti_adversarial.update(anti_adversarial_distance)
                    average_anti_genuine.update(anti_genuine_distance)
                    average_adversarial.update(adversarial_distance)

                    anti_genuine_distances.append(anti_genuine_distance)
                    anti_adversarial_distances.append(anti_adversarial_distance)

                    print('Average Anti-Adversarial: {:.3E} Average Anti-Genuine: {:.3E} Average Adversarial: {:.3E}'.format(average_anti_adversarial.avg, average_anti_genuine.avg, average_adversarial.avg))

                    #fixed_tpr = np.count_nonzero(np.array(anti_adversarial_distances) < minimum_distance) / len(anti_adversarial_distances)
                    #fixed_fpr = np.count_nonzero(np.array(anti_genuine_distances) < minimum_distance) / len(anti_genuine_distances)

                    #print('Fixed True Positive Rate: {:2.2f}% Fixed False Positive Rate: {:2.2f}%'.format(fixed_tpr * 100.0, fixed_fpr * 100.0))

                    plt.clf()

                    fig = plt.gcf()
                    fig.set_size_inches(18.5, 10.5, forward=True)

                    #plt.figure()

                    plt.subplot(2, 3, 1)
                    visualisation.plot_roc(-np.array(anti_genuine_distances), -np.array(anti_adversarial_distances))

                    plt.subplot(2, 3, 2)
                    visualisation.plot_histogram(np.array(anti_genuine_distances), 'blue', log_x=True)
                    visualisation.plot_histogram(np.array(anti_adversarial_distances), 'red', log_x=True)
                    visualisation.plot_histogram(np.concatenate([np.array(anti_genuine_distances), np.array(anti_adversarial_distances)]), 'green', log_x=True)

                    image = np.moveaxis(image, 0, -1)
                    adversarial = np.moveaxis(adversarial, 0, -1)
                    anti_adversarial = np.moveaxis(anti_adversarial, 0, -1)
                    anti_genuine = np.moveaxis(anti_genuine, 0, -1)

                    plt.subplot(2, 3, 3)
                    plt.title('Genuine ({})'.format(cifar_names[label]))
                    plt.imshow(image)  # division by 255 to convert [0, 255] to [0, 1]
                    plt.axis('off')

                    plt.subplot(2, 3, 4)
                    plt.title('Adversarial ({}, {:.2E})'.format(cifar_names[adversarial_label], adversarial_distance))
                    plt.imshow(adversarial)  # ::-1 to convert BGR to RGB
                    plt.axis('off')

                    plt.subplot(2, 3, 5)
                    plt.title('Anti-Genuine ({}, {:.2E})'.format(cifar_names[anti_genuine_label], anti_genuine_distance))
                    plt.imshow(anti_genuine)
                    plt.axis('off')

                    plt.subplot(2, 3, 6)
                    plt.title('Anti-Adversarial ({}, {:.2E})'.format(cifar_names[anti_adversarial_label], anti_adversarial_distance))
                    plt.imshow(anti_adversarial)
                    plt.axis('off')

                    plt.draw()
                    plt.pause(0.001)
            else:
                print('Attack failed')
        else:
            print('Misclassified')
