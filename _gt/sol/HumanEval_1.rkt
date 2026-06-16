#lang racket
(define (separate_paren_groups s)
  (let loop ([chars (filter (lambda (c) (not (char=? c #\space))) (string->list s))]
             [depth 0] [cur '()] [acc '()])
    (cond
      [(null? chars) (reverse acc)]
      [else
       (define c (car chars))
       (define nd (cond [(char=? c #\() (+ depth 1)]
                        [(char=? c #\)) (- depth 1)]
                        [else depth]))
       (define ncur (cons c cur))
       (if (and (char=? c #\)) (= nd 0))
           (loop (cdr chars) nd '() (cons (list->string (reverse ncur)) acc))
           (loop (cdr chars) nd ncur acc))])))
