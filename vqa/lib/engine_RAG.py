import time
import torch
from torch.autograd import Variable
import vqa.lib.utils as utils

# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

import re
import tqdm

def train(loader, model, retriever, tokenizer, criterion, optimizer, logger, epoch=0, print_freq=10):
    model.train ()
    meters = logger.reset_meters ('train')
    end = time.time()
    tq = tqdm.tqdm(total=len(loader))
    for i, sample in enumerate(loader):
        batch_size = sample['visual'].size(0)
        quries = [sample['question_query'][j] + " " + sample['question_raw'][j] for j in range(batch_size)]
        answers = [sample['sampled_answer'][j] for j in range(batch_size)]
        batch = tokenizer.prepare_seq2seq_batch (quries, answers, return_tensors="pt")
        source_ids, source_mask, target_ids = batch["input_ids"], batch["attention_mask"], batch["labels"]
        source_ids = source_ids.cuda()
        source_mask = source_mask.cuda()
        target_ids = target_ids.cuda()
        decoder_input_ids = target_ids[:, :-1].contiguous ()
        lm_labels = target_ids[:, 1 :].clone ()
        outputs = model (
            source_ids,
            attention_mask=source_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            labels=lm_labels,
            reduce_loss = True
        )
        loss = outputs['loss']
        optimizer.zero_grad ()
        loss.backward ()
        torch.cuda.synchronize ()
        optimizer.step ()
        torch.cuda.synchronize ()
        # compute output
        meters['loss'].update(loss.item(), n=batch_size)
        # measure accuracy and record loss
        meters['acc1'].update(-1, n=batch_size)
        meters['acc5'].update(-1, n=batch_size)

        # compute predictions for OpenEnded accuracy
        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()
        tq.update(1)
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {acc1.val:.3f} ({acc1.avg:.3f})\t'
                  'Acc@5 {acc5.val:.3f} ({acc5.avg:.3f})'.format(
                   epoch, i, len(loader),
                   batch_time=meters['batch_time'], data_time=meters['data_time'],
                   loss=meters['loss'], acc1=meters['acc1'], acc5=meters['acc5']))

            generated = model.generate (input_ids=source_ids, num_beams=2)
            ans = tokenizer.batch_decode (generated, skip_special_tokens=True)

@torch.no_grad()
def validate(loader, model, retriever, tokenizer, criterion, logger, epoch=0, print_freq=10):
    results = []

    # switch to evaluate mode
    model.eval()
    meters = logger.reset_meters('val')
    tq = tqdm.tqdm (total=len (loader))
    end = time.time()
    for i, sample in tqdm.tqdm(enumerate(loader)):
        batch_size = len(sample['question_raw'])
        target_answer  = Variable(sample['answer'].cuda(non_blocking=True))
        quries = [sample['question_query'][j] + " " + sample['question_raw'][j] for j in range (batch_size)]
        input_dict = tokenizer.prepare_seq2seq_batch (quries, return_tensors="pt")
        input_ids = input_dict["input_ids"].cuda()
        generated = model.generate (input_ids=input_ids, num_beams=2)
        ans = tokenizer.batch_decode (generated, skip_special_tokens=True)
        for j in range (batch_size) :
            d = {'question_id' : sample['question_id'][j].item (),
                 'answer' : ans[j].strip().lower(),
                 "true_answer" : loader.dataset.aid_to_ans[target_answer[j]],
                 'img_name' : sample['img_name'][j]}
            d['question_raw'] = sample['question_raw'][j]
            # d['answers_occurence'] = sample['answers_occurence'][j]
            results.append (d)
        if (i+1) % print_freq == 0:
            q = d['question_raw']
            a = d['answer']
            print("question: {} answer: {}\n".format(q, a))

        # compute output
        meters['loss'].update(-1, n=batch_size)
        # measure accuracy and record loss
        meters['acc1'].update(-1, n=batch_size)
        meters['acc5'].update(-1, n=batch_size)
        tq.update (1)

        # compute predictions for OpenEnded accuracy
        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()


    print(' * Acc@1 {acc1.avg:.3f} Acc@5 {acc5.avg:.3f}'
          .format(acc1=meters['acc1'], acc5=meters['acc1']))

    logger.log_meters('val', n=epoch)

    return meters['acc1'].avg, results


def test(loader, model, logger, epoch=0, print_freq=10):
    results = []
    testdev_results = []

    model.eval()
    meters = logger.reset_meters('test')

    end = time.time()
    for i, sample in enumerate(loader):
        batch_size = sample['visual'].size(0)
        input_visual   = Variable(sample['visual'].cuda(non_blocking=True), volatile=True)
        input_question = Variable(sample['question'].cuda(non_blocking=True), volatile=True)

        # compute output
        output = model(input_visual, input_question)

        # compute predictions for OpenEnded accuracy
        _, pred = output.data.cpu().max(1)
        pred.squeeze_()
        for j in range(batch_size):
            item = {'question_id': sample['question_id'][j].item(),
                    'answer': loader.dataset.aid_to_ans[pred[j]]}
            results.append(item)
            if sample['is_testdev'][j]:
                testdev_results.append(item)

        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                   i, len(loader), batch_time=meters['batch_time']))

    logger.log_meters('test', n=epoch)
    return results, testdev_results
