import foolbox


class AdyAttack():

    def __init__(self, model, max_iterations, num_classes, min, max):
        super(AdyAttack, self).__init__()
        fmodel = foolbox.models.PyTorchModel(model, bounds=(min, max),
                                             num_classes=num_classes)
        self.attack = foolbox.attacks.CarliniWagnerL2Attack(fmodel)
        self.max_iterations = max_iterations

    def carlini(self, original_image, true_class_id):
        adv_image = self.attack(original_image, true_class_id,
                           max_iterations=self.max_iterations)
        return adv_image


if __name__ == "__main__":
    print("Ady's attack")